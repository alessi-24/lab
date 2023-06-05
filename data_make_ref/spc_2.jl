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
gr()

#設定するパラメータ
#レプリカ数
const L = 100
#逆温度間隔決定
const gamma_temp = 1.3
#イテレーション数
const iter_num_burn = 70000
const iter_num = 50000
#ステップサイズ
const B20_C = 0.003
const B40_C = 0.003
const B44_C = 0.003
#ステップサイズ減少
const B20_d = 0.8
const B40_d = 0.8
const B44_d = 0.8

#書き出し
path = "/Users/nishimurarei/research/result_try2/spc/spc_2/"

#真のパラメータ値
function para_true()
    B20_true = 0.38
    B40_true = 0.16
    B44_true = 0.44
    return B20_true, B40_true, B44_true
end

#温度配列
function temp()
    Temp_table= collect(0.2:0.2:70) # length 350
    return Temp_table
end

#データ数
function data_num_spc()
    n = 350
    return n
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
    SpcHeat_Temp = zeros(n_spc)
    
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

function set_spc_noise()
    noise_spc = 1 / 10^2
    b_spc = 10^4
    return noise_spc, b_spc
end
const noise_spc, b_spc = set_spc_noise()

function making_data()
    #パラメータ真値
    B20_true, B40_true, B44_true = para_true()

    #シード値固定
    rng = MersenneTwister(1234)
    
    d_spc = Normal(0,noise_spc)
    SpcHeat_Temp = spc(Onn_make(B20_true,B40_true,B44_true))
    SpcHeat_Temp_noise = SpcHeat_Temp + rand(rng,d_spc,n_spc)
    
    return SpcHeat_Temp_noise
end
const SpcHeat_Temp_noise = making_data()

#逆温度
function make_beta(gamma,L)
    beta_list = gamma.^( (2:1:L) .- (L) )
    beta_list = pushfirst!(beta_list, 0.0)
    return beta_list
end
const beta_list = make_beta(gamma_temp, L)


#ステップサイズ
function step_size_make(C1,C2,d,e)
    #データ数
    n = data_num_spc()

    #一度低温領域のステップサイズをふる
    step_size_list = C1 ./ (n * beta_list).^ d
    
    #高温領域のステップサイズ
    step_size_list[n * beta_list .< e] .= C2
    
    return step_size_list
end
const step_size_B20 = step_size_make(B20_C,B20_C,B20_d,1)
const step_size_B40 = step_size_make(B40_C,B40_C,B40_d,1)
const step_size_B44 = step_size_make(B44_C,B44_C,B44_d,1)

#誤差関数
function error_spc(B20,B40,B44)
    error_value = sum((SpcHeat_Temp_noise - spc(Onn_make(B20,B40,B44))).^2)/(2*n_spc)
    return error_value
end

#コーシー分布
function cauchy(x)
    #パラメータ設定
    x_0 = 0
    gamma = 10
    
    return (1 / pi) * gamma / (gamma^2 + (x-x_0)^2)
end

#半コーシー分布
function half_cauchy(x)
    if x <= 0.0
        return 10^(-20)
    else
        #xが正のときはコーシー分布の値を返す
        x_0 = 0
        gamma = 10
        return (1 / pi) * gamma / (gamma^2 + (x-x_0)^2)
    end
end

#メトロポリス
#B20
function metropolis_spc_B20(para_B20_ini, para_B40_ini, para_B44_ini, flag, ac_B20)
    #ステップサイズ
    para_B20_renew = para_B20_ini + rand([1,-1],L) .* step_size_B20

    #事前分布
    prior_ratio = cauchy.(para_B20_renew) ./ cauchy.(para_B20_ini)
    
    #誤差関数の差
    error_dif = error_spc.(para_B20_renew, para_B40_ini, para_B44_ini) - error_spc.(para_B20_ini, para_B40_ini, para_B44_ini)

    #更新確率
    prob = exp.(- n_spc * b_spc * beta_list .* error_dif) .* prior_ratio
    
    #更新
    bit_vec = rand(L) .< prob
    bit_flip_vec = 1 .- bit_vec 
    para_B20_return = para_B20_renew .* bit_vec + para_B20_ini .* bit_flip_vec

    #交換率記録
    if flag
        ac_B20 += bit_vec
    end

    return para_B20_return, ac_B20
end

#B40
function metropolis_spc_B40(para_B20_ini, para_B40_ini, para_B44_ini, flag, ac_B40)
    #ステップサイズ
    para_B40_renew = para_B40_ini + rand([1,-1],L) .* step_size_B40

    #事前分布
    prior_ratio = cauchy.(para_B40_renew) ./ cauchy.(para_B40_ini)
    
    #誤差関数の差
    error_dif = error_spc.(para_B20_ini, para_B40_renew, para_B44_ini) - error_spc.(para_B20_ini, para_B40_ini, para_B44_ini)

    #更新確率
    prob = exp.(- n_spc * b_spc * beta_list .* error_dif) .* prior_ratio
    
    #更新
    bit_vec = rand(L) .< prob
    bit_flip_vec = 1 .- bit_vec 
    para_B40_return = para_B40_renew .* bit_vec + para_B40_ini .* bit_flip_vec

    if flag
        ac_B40 += bit_vec
    end

    return para_B40_return, ac_B40
end

#B44
function metropolis_spc_B44(para_B20_ini, para_B40_ini, para_B44_ini, flag, ac_B44)
    #ステップサイズ
    para_B44_renew = para_B44_ini + rand([1,-1],L) .* step_size_B44

    #事前分布
    prior_ratio = half_cauchy.(para_B44_renew) ./ half_cauchy.(para_B44_ini)
    
    #誤差関数の差
    error_dif = error_spc.(para_B20_ini, para_B40_ini, para_B44_renew) - error_spc.(para_B20_ini, para_B40_ini, para_B44_ini)

    #更新確率
    prob = exp.(- n_spc * b_spc * beta_list .* error_dif) .* prior_ratio
    
    #更新
    bit_vec = rand(L) .< prob
    bit_flip_vec = 1 .- bit_vec 
    para_B44_return = para_B44_renew .* bit_vec .+ para_B44_ini .* bit_flip_vec

    if flag
        ac_B44 += bit_vec
    end

    return para_B44_return, ac_B44
end

function metropolis_spc(para_B20_ini, para_B40_ini, para_B44_ini, flag, ac_B20, ac_B40, ac_B44)
    para_B20_renew, ac_B20 = metropolis_spc_B20(para_B20_ini, para_B40_ini, para_B44_ini, flag, ac_B20)
    para_B40_renew, ac_B40 = metropolis_spc_B40(para_B20_ini, para_B40_ini, para_B44_ini, flag, ac_B40)
    para_B44_renew, ac_B44 = metropolis_spc_B44(para_B20_ini, para_B40_ini, para_B44_ini, flag, ac_B44)
    return para_B20_renew, para_B40_renew, para_B44_renew, ac_B20, ac_B40, ac_B44
end

#レプリカ交換
function exchange_spc(para_B20, para_B40, para_B44, flag, ex_rate)
    #逆温度の差
    beta_dif = beta_list[2:end] - beta_list[1:end-1]

    @inbounds for replica in 1:L-1
        #交換確率
        prob = exp(
            n_spc * b_spc * beta_dif[replica] * (
                error_spc(para_B20[replica + 1], para_B40[replica + 1], para_B44[replica + 1]) - error_spc(para_B20[replica], para_B40[replica], para_B40[replica])
                )
            )

        #ビット
        bit = rand() < prob # 1で交換、0で非交換

        if bit == 1 #交換
            para_B20[replica+1], para_B20[replica] = para_B20[replica], para_B20[replica+1]
            para_B40[replica+1], para_B40[replica] = para_B40[replica], para_B40[replica+1]
            para_B44[replica+1], para_B44[replica] = para_B44[replica], para_B44[replica+1]
        end

        if flag
            ex_rate[replica] += bit
        end
    end

    return para_B20, para_B40, para_B44, ex_rate
end

#事前分布による初期パラメータ決定
function initial_para_make()
    #パラメータ候補値
    para_num = collect(0.01:0.01:1.00)
    #確率
    para_prob = cauchy.(para_num)

    return sample(para_num, ProbabilityWeights(para_prob),L)
end

function initial_para_make_around_true()
    B20_true, B40_true, B44_true = para_true()

    #パラメータ候補値
    B20_para = ones(Int16,L) * B20_true
    B40_para = ones(Int16,L) * B40_true
    B44_para = ones(Int16,L) * B44_true

    return B20_para, B40_para, B44_para
end

function exmc(iter_num_burn, iter_num)
    #初期パラメータ設定
    para_B20 = initial_para_make() 
    para_B40 = initial_para_make() 
    para_B44 = initial_para_make() 

    #para_B20, para_B40, para_B44 = initial_para_make_around_true()

    #mcmcの中で交換する割合
    exchange_iter = 3

    #採択率・交換率
    ac_B20 = zeros(Float16, L)
    ac_B40 = zeros(Float16, L)
    ac_B44 = zeros(Float16, L)
    ex_rate = zeros(Float16, L)

    #パラメータ
    B20_save = zeros(Float16, iter_num, L)
    B40_save = zeros(Float16, iter_num, L)
    B44_save = zeros(Float16, iter_num, L)

    #burn in
    flag_burn = false
    @inbounds for iter in 1:iter_num_burn
        para_B20, para_B40, para_B44, _, _, _ = metropolis_spc(para_B20, para_B40, para_B44, flag_burn, 0, 0, 0)
        
        para_B20, para_B40, para_B44, _ = exchange_spc(para_B20, para_B40, para_B44, flag_burn, 0)
        
    end

    #after burn in
    #burn in
    flag = true
    @inbounds for iter in 1:iter_num
        para_B20, para_B40, para_B44, ac_B20, ac_B40, ac_B44 = metropolis_spc(para_B20, para_B40, para_B44, flag, ac_B20, ac_B40, ac_B44)
        
        
        para_B20, para_B40, para_B44, ex_rate = exchange_spc(para_B20, para_B40, para_B44, flag, ex_rate)
        

        B20_save[iter,:] = para_B20
        B40_save[iter,:] = para_B40
        B44_save[iter,:] = para_B44

    end

    ac_B20 /= iter_num
    ac_B40 /= iter_num
    ac_B44 /= iter_num
    ex_rate /= iter_num

    return B20_save, B40_save, B44_save, ac_B20, ac_B40, ac_B44, ex_rate
end

para_B20, para_B40, para_B44, ac_B20, ac_B40, ac_B44, ex_rate = exmc(iter_num_burn, iter_num)

#パラメータ設定
df_const = DataFrame(
    replica = L,
    gamma=gamma_temp,
    iter=[iter_num_burn, iter_num,0],
    ID = ["B20", "B40", "B44"],
    step = [B20_C, B40_C, B44_C],
    step_d = [B20_d, B40_d,B44_d]
)

#採択率・交換率
df_ac_ex = DataFrame(B20 = ac_B20, B40 = ac_B40, B44 = ac_B44, ex = ex_rate)

#パラメータ
df_para_B20 = DataFrame(para_B20, :auto)
df_para_B40 = DataFrame(para_B40, :auto)
df_para_B44 = DataFrame(para_B44, :auto)

df_const |> CSV.write(path * "const.csv",writeheader=true)
df_ac_ex |> CSV.write(path * "ac_ex.csv",writeheader=true)
df_para_B20 |> CSV.write(path * "B20.csv",writeheader=true)
df_para_B40 |> CSV.write(path * "B40.csv",writeheader=true)
df_para_B44 |> CSV.write(path * "B44.csv",writeheader=true)