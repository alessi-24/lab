#package
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

function main()
    # 読み込み
    read_file_path = "/Users/test/home/lab_research_1/data_make/data/chi/chi.h5"
    chi_Temp = read_file(read_file_path)

    # 逆数にする
    chi_Temp_inv = chi_inv_make(chi_Temp)

    # 書き込み
    write_file_path = "/Users/test/home/lab_research_1/data_make/data/chi_inv/chi_inv.h5"
    write_file(write_file_path, chi_Temp_inv)
end


main()

