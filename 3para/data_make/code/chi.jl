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

#真のパラメータ値
function para_true()
    B20_true = 0.381
    B40_true = 0.165
    B44_true = 0.447
    return B20_true, B40_true, B44_true
end

#温度配列
function temp()
    Temp_table = collect(0.2:0.2:70) # length 350
    return Temp_table
end

#各種結晶場パラメータ
function ini()
    # Hund's Rule Ground J-Multiplet Ce3+ n4f=1
    n4f = 1.0
    L = 3.0
    S = 0.5
    J = L - S
    g = 1.0 + (J * (J + 1.0) + S * (S + 1.0) - L * (L + 1.0)) / (2.0 * J * (J + 1.0))
    Jz = [2.5, 1.5, 0.5, -0.5, -1.5, -2.5]
    return J, g, Jz
end



#結晶場行列
function Onn_make(B20,B40,B44)
    # O20
    O20_vec_x = [10.0, -2.0, -8.0, -8.0, -2.0, 10.0]
    O20_vec = O20_vec_x * B20
    O20 = diagm(0 => O20_vec)
    
    # O40
    O40_vec_x = [60.0, -180.0, 120.0, 120.0, -180.0, 60.0]
    O40_vec = O40_vec_x * B40
    O40 = diagm(0 => O40_vec)

    Onn = O20 + O40
    
    # O44
    O44_value = sqrt(120.0 * 24.0) * B44 / 2.0
    Onn[5,1] = O44_value
    Onn[6,2] = O44_value
    Onn[1,5] = O44_value
    Onn[2,6] = O44_value
    
    return Onn
end

#磁化率データ作成
function chi(Onn)
    # 温度定義
    Temp_table_chi = temp()

    _, g, Jz = ini()
    magfield = 0.01
    #magdir = [0,0,1]
    #nor_magdir = [0,0,1]
    
    # 対角要素
    Hmag_vec_0 = Jz * magfield * g * 0.67171
    Hmag = diagm(0 => Hmag_vec_0)
    
    # 非対角要素はmagdir[x,y]が0なので0
    
    # 結晶場＋磁場ハミルトニアンの行列要素
    H = Onn + Hmag
    
    eigval, eigvec = eigen(H)
    chi_inv_Temp = zeros(length(Temp_table_chi))
    
    @inbounds for (i, Temp) in enumerate(Temp_table_chi)
        eigval_2 = - eigval / Temp
        eigval_2_max = maximum(eigval_2)
        eigval_ratio = eigval_2 .- eigval_2_max
        exp_eigval = exp.(eigval_ratio)
        sumexp = sum(exp_eigval)

        mag_z = zeros(6)
        @simd for j in 1:6
            eigvec_check = eigvec[:,j]
            mag_z[j] = eigvec_check' * (eigvec_check .* Jz) * exp_eigval[j] / sumexp
        end

        Jmag_z = sum(mag_z) * g * (-1)
        Jmag = Jmag_z

        chi = Jmag / magfield * 0.5585
        chi_inv = 1.0 / chi
        chi_inv_Temp[i] = chi_inv
        #chi_Temp[i] = chi
    end
    return chi_inv_Temp
end

# ファイル書き込み
function write_file(file_path, data)
    h5open(file_path, "w") do file #ファイルtest.h5を開き、書き込み
        write(file,"chi",data)
    end
end


function main()
    B20_true,B40_true,B44_true = para_true()
    chi_Temp = chi(Onn_make(B20_true,B40_true,B44_true))

    file_path = "/Users/nishimura/home/lab/3para/data_make/data/chi/chi.h5"
    write_file(file_path, chi_Temp)
end


main()