using Distributions
using Random
using LinearAlgebra
using Plots
using StatsBase
using LaTeXStrings
using DataFrames
using CSV
using Base.Threads
using HDF5
gr()

function read_file(file_path)
    file = h5open(file_path, "r")
    chi_Temp = read(file, "chi")
    return chi_Temp
end

function write_file(file_path, data)
    h5open(file_path, "w") do file #ファイルtest.h5を開き、書き込み
    write(file,"chi_inv",data)
    end
end

function chi_inv_make(data)
    data_inv = (data).^(-1)
    return data_inv
end

function add_noise(data, noise)
    n = length(data)
    d_chi = Normal(0,noise)
    rng = MersenneTwister(1234)
    data_noise = data + rand(rng, d_chi,n)
    return data_noise
end

function make_noise_data(noise)
    # 読み込み
    read_file_path = "/Users/test/home/lab_research_1/data_make/data/chi/chi.h5"
    chi_Temp = read_file(read_file_path)

    # ノイズデータ
    chi_Temp_inv = chi_inv_make(chi_Temp)
    chi_Temp_noise_inv = add_noise(chi_Temp_inv, noise)

    return chi_Temp_noise_inv
end

function main()
    # ノイズのリスト
    noise_list = [10^(-1), 10^(-0.5), 10^(0), 10^(0.5), 10^(1)]

    # ノイズ入りデータ作成
    for (index, noise) in enumerate(noise_list)
        chi_Temp_noise_inv = make_noise_data(noise)
        
        # 書き込みpath
        write_file_path = "/Users/test/home/lab_research_1/data_make/data/chi_inv/chi_inv_" * string(index) * ".h5"

        # 書き込み
        write_file(write_file_path, chi_Temp_noise_inv)
    end
end

main()