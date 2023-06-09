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
    data = read(file, "spc")
    return data
end

function write_file(file_path, data)
    h5open(file_path, "w") do file #ファイルtest.h5を開き、書き込み
    write(file,"spc",data)
    end
end


function add_noise(data, noise)
    n = length(data)
    d = Normal(0,noise)
    rng = MersenneTwister(1234)
    data_noise = data + rand(rng, d, n)
    return data_noise
end

function make_noise_data(noise)
    # 読み込み
    read_file_path = "/Users/nishimura/home/lab/data_make/data/spc/spc.h5"
    SpcHeat_Temp = read_file(read_file_path)

    # ノイズデータ
    SpcHeat_Temp_noise = add_noise(SpcHeat_Temp, noise)

    return SpcHeat_Temp_noise
end

function main()
    # ノイズのリスト
    noise_list = [10^(-2.5), 10^(-2), 10^(-1.5), 10^(-1), 10^(-0.5)]

    # ノイズ入りデータ作成
    for (index, noise) in enumerate(noise_list)
        SpcHeat_Temp_noise = make_noise_data(noise)
        
        # 書き込みpath
        write_file_path = "/Users/test/home/lab_research_1/data_make/data/spc/spc_" * string(index) * ".h5"

        # 書き込み
        write_file(write_file_path, SpcHeat_Temp_noise)
    end
end

function main2()
    # ノイズのリスト
    noise_list = [10^(0.0), 10^(0.5), 10^(1.0)]

    # ノイズ入りデータ作成
    for (index, noise) in enumerate(noise_list)
        SpcHeat_Temp_noise = make_noise_data(noise)
        
        # 書き込みpath
        index_up = index + 5
        write_file_path = "/Users/nishimura/home/lab/data_make/data/spc/spc_" * string(index_up) * ".h5"

        # 書き込み
        write_file(write_file_path, SpcHeat_Temp_noise)
    end
end

main2()