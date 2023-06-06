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

#設定するパラメータ
#レプリカ数
L = 72
#逆温度間隔決定
gamma_temp = 1.4
#イテレーション数
iter_num_burn = 500
iter_num = 1000
#ステップサイズ
B40_C = 0.01
#ステップサイズ減少
B40_d = 0.6
# ノイズの大きさ
b_spc = 10^(1)

#読み込みのファイルパス
read_file_path = "/Users/nishimura/home/lab/data_make/data/spc/spc_5.h5"

#書き出し
path = "/Users/nishimura/home/lab/exmc/result/spc/spc5_"


#真のパラメータ値
function para_true()
    B40_true = 0.1167
    return B40_true
end

#温度配列
function temp()
    Temp_table_spc = collect(0.2:0.2:70) # length 350
    return Temp_table_spc
end

#データ数
function data_num_spc()
    n_spc = 350
    return n_spc
end
const n_spc = data_num_spc()

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
function Onn_make(B40)
    # O40
    O40_vec_x = [60.0, -180.0, 120.0, 120.0, -180.0, 60.0]
    O40_vec = O40_vec_x * B40
    O40 = diagm(0 => O40_vec)
    Onn = O40
    # O44
    B44 = 5 * B40
    O44_value = sqrt(120.0 * 24.0) * B44 / 2.0
    Onn[5,1] = O44_value
    Onn[6,2] = O44_value
    Onn[1,5] = O44_value
    Onn[2,6] = O44_value
    return Onn
end

# 比熱
function spc(Onn)
    #パラメータ読み込み
    _, g, _ = ini()

    # 非対角要素
    Hmag_vec_1 = sqrt.([5,8,9,8,5]) * (1 + 1*im) * 5 * g * 0.67171 / 2
    Hmag_1 = diagm(1 => Hmag_vec_1)
    Hmag_2 = diagm(-1 => conj.(Hmag_vec_1))
    Hmag = Hmag_1 + Hmag_2

    #結晶場＋磁場ハミルトニアンの行列要素
    H = Onn + Hmag

    eigval, eigvec = eigen(H)

    Temp_table_spc = temp()
    SpcHeat_Temp = zeros(length(Temp_table_spc))
    
    @inbounds for (i, Temp) in enumerate(Temp_table_spc)
        eigval_2 = - eigval / Temp
        eigval_2_max = maximum(eigval_2)
        eigval_ratio = eigval_2 .- eigval_2_max
        exp_eigval = exp.(eigval_ratio)

        Z0 = sum(exp_eigval)
        Z1 = sum(eigval_2 .* exp_eigval)
        Z2 = sum(eigval_2.^2 .* exp_eigval)
        
        SpcHeat=(- (Z1/Z0)^2 + (Z2/Z0) )*8.31441
        SpcHeat_Temp[i] = SpcHeat
    end
    return SpcHeat_Temp
end

# データ読み込み
function read_file(file_path)
    file = h5open(file_path, "r")
    SpcHeat_Temp = read(file, "spc")
    return SpcHeat_Temp
end
SpcHeat_Temp_noise = read_file(read_file_path)

# 逆温度
function make_beta(gamma,L)
    #最初は等差で決定
    beta_list = gamma.^( (2:1:L) .- (L) )
    beta_list = pushfirst!(beta_list, 0.0)
    return beta_list
end
beta_list = make_beta(gamma_temp, L)

#ステップサイズ
function step_size_make(C,d)
    #データ数
    n = 350

    #一度低温領域のステップサイズをふる
    step_size_list = C ./ (n * beta_list).^ d
    
    #高温領域のステップサイズ
    step_size_list[n * beta_list .< 1] .= C
    
    return step_size_list
end
step_size_B40 = step_size_make(B40_C,B40_d)


#誤差関数
function error_spc(B40)
    error_value = sum((SpcHeat_Temp_noise - spc(Onn_make(B40))).^2)/(2*n_spc)
    return error_value
end


#一様分布
function prior(x)
    if x >= -1 && x <= 1
        return 0.5
    else
        return 10^(-5)
    end
end


#メトロポリス
#B40
function metropolis_spc_B40_burn(para_B40_ini)
    #ステップサイズ
    para_B40_renew = para_B40_ini + rand(L) .* rand([1,-1],L) .* step_size_B40

    #事前分布
    prior_ratio = prior.(para_B40_renew) ./ prior.(para_B40_ini)
    
    # 誤差関数の差
    # これを並列化する
    error_dif = zeros(L)
    #error_dif = error_spc.(para_B40_renew) - error_spc.(para_B40_ini)
    @threads for l in 1:L
        error_dif[l] = error_spc(para_B40_renew[l]) - error_spc(para_B40_ini[l])
    end

    #更新確率
    prob = exp.(- n_spc * b_spc * beta_list .* error_dif) .* prior_ratio
    
    #更新
    bit_vec = rand(L) .< prob
    bit_flip_vec = 1 .- bit_vec 
    para_B40_return = para_B40_renew .* bit_vec + para_B40_ini .* bit_flip_vec

    return para_B40_return
end

function metropolis_spc_B40(para_B40_ini, ac_B40)
    #ステップサイズ
    para_B40_renew = para_B40_ini + rand(L) .* rand([1,-1],L) .* step_size_B40

    #事前分布
    prior_ratio = prior.(para_B40_renew) ./ prior.(para_B40_ini)
    
    # 誤差関数の差
    # これを並列化する
    error_dif = zeros(L)
    #error_dif = error_spc.(para_B40_renew) - error_spc.(para_B40_ini)
    @threads for l in 1:L
        error_dif[l] = error_spc(para_B40_renew[l]) - error_spc(para_B40_ini[l])
    end

    #更新確率
    prob = exp.(- n_spc * b_spc * beta_list .* error_dif) .* prior_ratio
    
    #更新
    bit_vec = rand(L) .< prob
    bit_flip_vec = 1 .- bit_vec 
    para_B40_return = para_B40_renew .* bit_vec + para_B40_ini .* bit_flip_vec

    #交換率記録
    ac_B40 += bit_vec

    return para_B40_return, ac_B40
end


#レプリカ交換
function exchange_spc_burn(para_B40)
    #逆温度の差
    beta_dif = beta_list[2:end] - beta_list[1:end-1]

    for replica in 1:L-1
        #交換確率
        prob = exp(
            n_spc * b_spc * beta_dif[replica] * (
                error_spc(para_B40[replica + 1]) - error_spc(para_B40[replica])
                )
            )

        #ビット
        bit = rand() < prob # 1で交換、0で非交換

        if bit == 1 #交換
            para_B40[replica+1], para_B40[replica] = para_B40[replica], para_B40[replica+1]
        end

    end

    return para_B40
end

function exchange_spc(para_B40, ex_rate)
    #逆温度の差
    beta_dif = beta_list[2:end] - beta_list[1:end-1]

    for replica in 1:L-1
        #交換確率
        prob = exp(
            n_spc * b_spc * beta_dif[replica] * (
                error_spc(para_B40[replica + 1]) - error_spc(para_B40[replica])
                )
            )

        #ビット
        bit = rand() < prob # 1で交換、0で非交換

        if bit == 1 #交換
            para_B40[replica+1], para_B40[replica] = para_B40[replica], para_B40[replica+1]
        end

        ex_rate[replica] += bit
    end

    return para_B40, ex_rate
end


#事前分布による初期パラメータ決定
function initial_para_make()
    #パラメータ候補値
    para_num = collect(0.0:0.01:0.2)
    #確率
    para_prob = prior.(para_num)
    return sample(para_num, ProbabilityWeights(para_prob),L)
end


function exmc(iter_num_burn, iter_num)
    #初期パラメータ設定
    para_B40 = initial_para_make() 

    #採択率・交換率
    ac_B40 = zeros(L)
    ex_rate = zeros(L)

    #パラメータ
    save_size = iter_num * 1
    B40_save = zeros(save_size, L)

    
    # burn in
    # print
    print_num_burn = Int16(iter_num_burn / 10)
    for iter in 1:iter_num_burn
        para_B40 = metropolis_spc_B40_burn(para_B40)
        
        if iter % 2 == 0
            para_B40 = exchange_spc_burn(para_B40)
        end

        if iter % print_num_burn == 0
            iter_perent = iter / print_num_burn * 10
            println("finish $iter_perent percent (burn in)")
        end
    end

    # after burn in
    # print
    print_num = Int16(iter_num / 10)
    for iter in 1:iter_num
        para_B40, ac_B40 = metropolis_spc_B40(para_B40, ac_B40)
        B40_save[iter,:] = para_B40

        if iter % 2 == 0
            para_B40, ex_rate = exchange_spc(para_B40, ex_rate)
        end

        if iter % print_num == 0
            iter_perent = iter / print_num * 10
            println("finish $iter_perent percent")
        end
    end

    ac_B40 /= iter_num
    ex_rate /= (iter_num / 2)

    return B40_save, ac_B40, ex_rate
end

para_B40, ac_B40, ex_rate = exmc(iter_num_burn, iter_num)


# rate保存
ac_ex_path = path * "ac_ex.h5"
h5open(ac_ex_path, "w") do file
    write(file, "ac", ac_B40)
    write(file, "ex", ex_rate)
end


# パラメータ保存
B40_path = path * "B40.h5"
h5open(B40_path, "w") do file
    write(file, "B40", para_B40)
end
