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

a = collect(1:7)



#=
@time h5open("/Users/test/home/lab_research_1/data_make/code/test2.h5", "w") do file #ファイルtest.h5を開き、書き込み
    write(file,"a",a) 
end
=#


file_o = h5open("/Users/test/home/lab_research_1/data_make/code/test2.h5", "r")

a_o = read(file_o,"a")
close(file_o)

println(a_o)
