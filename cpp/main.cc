#include <iostream>
#include <memory>
#include <vector>
#include <cmath>

#include "vart/runner.hpp"
#include "vart/runner_helper.hpp"
#include <xir/graph/graph.hpp>
#include <xir/tensor/tensor.hpp>
#include <xir/util/data_type.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

std::vector<uint8_t> tensor_from_stbimage(const std::string& filename) {
    int x, y, n;
    uint8_t* rawptr = stbi_load(filename.c_str(), &x, &y, &n, 0);
    std::cout << "x, y, n: " << x << " " << y << " " << n << std::endl;
    int pxs = x * y * n;
    std::vector<uint8_t> vec(pxs);
    memcpy(vec.data(), rawptr, pxs);
    return vec;
}

float get_scale(const xir::Tensor* tensor) {
    int fixpos = tensor->get_attr<int>("fix_point");
    return std::exp2f(1.0f * (float)fixpos);
}

void print_softmax(int8_t* data, size_t len, float scale) {
    double denom = 0;
    std::vector<double> vec;
    for (int i = 0; i < len; ++i) {
        float tmp = static_cast<double>((data[i] / 128.0) * scale);
        tmp = std::exp(tmp);
        denom += tmp;
        vec.push_back(tmp);
    }

    std::cout << std::endl;
    for (int x = 0; x < len; ++x) {
        std::cout << (vec[x] / denom) << " ";
    }
    std::cout << std::endl;
}


int main(int argc, char *argv[]) {
    static constexpr const char* MODELNAME = "quantize_result/deploy.xmodel";
    auto image = tensor_from_stbimage(argv[1]);

    auto graph = xir::Graph::deserialize(MODELNAME);

    auto graph_root = graph.get()->get_root_subgraph();
    auto children = graph_root->children_topological_sort();

    // Check subgraphs on CLI with `xdputil xmodel quantize_result/deploy.xmodel -l`
    const xir::Subgraph* subgraph;
    for (auto c : children) {
        auto device = c->get_attr<std::string>("device");
        std::cout << "Device: " << device << std::endl;
        std::cout << c->get_name() << std::endl;
        if (device == "DPU") {
            subgraph = c;
        }
    }
    auto runner = vart::Runner::create_runner(subgraph, "run");
    auto input_tensors = runner->get_input_tensors();
    float in_scale = get_scale(input_tensors[0]);
    auto output_tensors = runner->get_output_tensors();
    float out_scale = get_scale(output_tensors[0]);
    std::cout << "Input scale " << in_scale << std::endl;
    std::cout << "Output scale " << out_scale << std::endl;

    std::cout << "Populating input tensor" << std::endl;
    int input_num = input_tensors.size();
    std::cout << input_num << " Input(s)" << std::endl;
    auto input_tensor_buffers = std::vector<std::unique_ptr<vart::TensorBuffer>>();
    for (auto input : input_tensors) {
        auto t = vart::alloc_cpu_flat_tensor_buffer(input);

        // Copy data by ptr into the buffer
        auto tensor_data = t->data();
        // This CpuFlatTensorBuffer class returns a uint64_t by reinterpret casting its 64bit data
        // pointer into an integer, wtf? Need to undo that here to get something we can use as a ptr
        int8_t* raw_ptr_int = reinterpret_cast<int8_t*>(std::get<0>(tensor_data));
        size_t raw_bytes = std::get<1>(tensor_data);
        std::cout << "Input tensor is " << raw_bytes << " bytes long" << std::endl;
        for (int x = 0; x < raw_bytes; ++x) {
            // In our training code, the uint8 images are squished to the range (0, 1)
            // before being input to the network, so replicate that here.
            float tmp = static_cast<float>(image[x]) / 256;

            raw_ptr_int[x] = static_cast<int8_t>(tmp * in_scale);
        }

        input_tensor_buffers.emplace_back(std::move(t));
    }

    int output_num = output_tensors.size();
    std::cout << output_num << " Output(s)" << std::endl;
    auto output_tensor_buffers = std::vector<std::unique_ptr<vart::TensorBuffer>>();
    for (auto output : output_tensors) {
        auto t = vart::alloc_cpu_flat_tensor_buffer(output);
        output_tensor_buffers.emplace_back(std::move(t));
    }

    // sync input tensor buffers
    for (auto& input : input_tensor_buffers) {
        input->sync_for_write(0, input->get_tensor()->get_data_size() /
                input->get_tensor()->get_shape()[0]);
    }
    // run runner
    std::cout << "Executing runner..." << std::endl;
    // This is really weird; I hold input_tensor_buffers as a vec of unique ptrs, but the
    // execute_runner function we need to use it with needs a vec of raw ptrs, requiring
    // us to call get(). The example actually calls get() when populating the input_tensor_buffers
    // vec, but that looks wrong b/c the unique_ptr it called get() on goes out of scope in the loop
    std::vector<vart::TensorBuffer*> input_ptrs;
    for (auto& ptr : input_tensor_buffers) {
        input_ptrs.push_back(ptr.get());
    }
    std::vector<vart::TensorBuffer*> output_ptrs;
    for (auto& ptr : output_tensor_buffers) {
        output_ptrs.push_back(ptr.get());
    }

    auto v = runner->execute_async(input_ptrs, output_ptrs);
    auto status = runner->wait((int)v.first, 1000000000);

    // sync output tensor buffers
    for (auto& output : output_tensor_buffers) {
        output->sync_for_read(0, output->get_tensor()->get_data_size() /
        output->get_tensor()->get_shape()[0]);
    }

    auto& out_buf = output_tensor_buffers.back();
    auto out_data = out_buf->data();
    int8_t* raw_ptr_int = reinterpret_cast<int8_t*>(std::get<0>(out_data));
    size_t raw_bytes = std::get<1>(out_data);
    std::cout << "Output tensor is " << raw_bytes << " bytes long" << std::endl;
    print_softmax(raw_ptr_int, raw_bytes, out_scale);

    return 0;
}
