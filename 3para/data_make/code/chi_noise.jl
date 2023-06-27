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
    data = read(file, "chi")
    return data
end

function write_file(file_path, data)
    h5open(file_path, "w") do file #ファイルtest.h5を開き、書き込み
    write(file,"chi",data)
    end
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
    read_file_path = "/Users/nishimura/home/lab/3para/data_make/data/chi/chi.h5"
    chi_Temp_inv = read_file(read_file_path)

    # ノイズデータ
    chi_Temp_noise_inv = add_noise(chi_Temp_inv, noise)

    return chi_Temp_noise_inv
end

function main()
    # ノイズのリスト
    noise_list = [10^(-3.0), 10^(-2.5), 10^(-2.0), 10^(-1.5), 10^(-1.0), 10^(-0.5), 10^(0.0), 10^(0.5), 10^(1.0)]

    # ノイズ入りデータ作成
    for (index, noise) in enumerate(noise_list)
        chi_Temp_noise_inv = make_noise_data(noise)
        
        # 書き込みpath
        write_file_path = "/Users/nishimura/home/lab/3para/data_make/data/chi/chi_" * string(index) * ".h5"

        # 書き込み
        write_file(write_file_path, chi_Temp_noise_inv)
    end
end

main()