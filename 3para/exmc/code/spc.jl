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
gamma_temp = 1.2
#イテレーション数
iter_num_burn = 1000
iter_num = 1000
#ステップサイズ
B20_C = 0.001
B40_C = 0.001
B44_C = 0.001
#ステップサイズ減少
B20_d = 0.7
B40_d = 0.7
B44_d = 0.7
# ノイズの大きさ
b_spc = 10^(6)

#読み込みのファイルパス
read_file_path = "/Users/nishimura/home/lab/3para/data_make/data/spc/spc_3.h5"

#書き出し
path = "/Users/nishimura/home/lab/3para/exmc/result/spc/spc3_"



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

#データ数
function data_num()
    n = 350
    return n
end
const n = data_num()

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
function read_file_spc(file_path)
    file = h5open(file_path, "r")
    SpcHeat_Temp = read(file, "spc")
    return SpcHeat_Temp
end
SpcHeat_Temp_noise = read_file_spc(read_file_path)

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
step_size_B20 = step_size_make(B20_C,B20_d)
step_size_B40 = step_size_make(B40_C,B40_d)
step_size_B44 = step_size_make(B44_C,B44_d)

#誤差関数
function error_spc(B20,B40,B44)
    error_value = sum((SpcHeat_Temp_noise - spc(Onn_make(B20,B40,B44))).^2)/(2*n)
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

#パラメータ更新
function para_renew(para_B20_ini,para_B40_ini,para_B44_ini)
    # 更新
    # 事前分布は各パラメータの積なので、更新したパラメータのみ考える
    para_renew_number = rand([1,2,3])
    if para_renew_number == 1
        para_B20_renew = para_B20_ini + rand(L) .* rand([1,-1],L) .* step_size_B20
        prior_ratio = prior.(para_B20_renew) ./ prior.(para_B20_ini)
        return para_B20_renew,para_B40_ini,para_B44_ini,prior_ratio,para_renew_number
    elseif para_renew_number == 2
        para_B40_renew = para_B40_ini + rand(L) .* rand([1,-1],L) .* step_size_B40
        prior_ratio = prior.(para_B40_renew) ./ prior.(para_B40_ini)
        return para_B20_ini,para_B40_renew,para_B44_ini,prior_ratio,para_renew_number
    elseif para_renew_number == 3
        para_B44_renew = para_B44_ini + rand(L) .* rand([1,-1],L) .* step_size_B44
        prior_ratio = prior.(para_B44_renew) ./ prior.(para_B44_ini)
        return para_B20_ini,para_B40_ini,para_B44_renew,prior_ratio,para_renew_number
    end
end

#メトロポリス更新
function metropolis_spc_burn(para_B20_ini,para_B40_ini,para_B44_ini)
    para_B20_renew,para_B40_renew,para_B44_renew,prior_ratio,para_renew_number = para_renew(para_B20_ini,para_B40_ini,para_B44_ini)
    
    # 誤差関数の差
    error_dif = zeros(L)
    @threads for l in 1:L
        error_dif[l] = error_spc(para_B20_renew[l],para_B40_renew[l],para_B44_renew[l]) - error_spc(para_B20_ini[l],para_B40_ini[l],para_B44_ini[l])
    end

    #更新確率
    prob = exp.(- n * b_spc * beta_list .* error_dif) .* prior_ratio
    
    #更新
    bit_vec = rand(L) .< prob
    bit_flip_vec = 1 .- bit_vec

    if para_renew_number == 1
        para_B20_return = para_B20_renew .* bit_vec + para_B20_ini .* bit_flip_vec
        return para_B20_return, para_B40_ini, para_B44_ini
    elseif para_renew_number == 2
        para_B40_return = para_B40_renew .* bit_vec + para_B40_ini .* bit_flip_vec
        return para_B20_ini, para_B40_return, para_B44_ini
    elseif para_renew_number == 3
        para_B44_return = para_B44_renew .* bit_vec + para_B44_ini .* bit_flip_vec
        return para_B20_ini, para_B40_ini, para_B44_return
    end
    
end

#メトロポリス更新
function metropolis_spc(para_B20_ini,para_B40_ini,para_B44_ini,ac)
    para_B20_renew,para_B40_renew,para_B44_renew,prior_ratio,para_renew_number = para_renew(para_B20_ini,para_B40_ini,para_B44_ini)
    
    # 誤差関数の差
    error_dif = zeros(L)
    @threads for l in 1:L
        error_dif[l] = error_spc(para_B20_renew[l],para_B40_renew[l],para_B44_renew[l]) - error_spc(para_B20_ini[l],para_B40_ini[l],para_B44_ini[l])
    end

    #更新確率
    prob = exp.(- n * b_spc * beta_list .* error_dif) .* prior_ratio
    
    #更新
    bit_vec = rand(L) .< prob
    bit_flip_vec = 1 .- bit_vec

    if para_renew_number == 1
        para_B20_return = para_B20_renew .* bit_vec + para_B20_ini .* bit_flip_vec
        ac[:,1] += bit_vec
        return para_B20_return, para_B40_ini, para_B44_ini, ac
    elseif para_renew_number == 2
        para_B40_return = para_B40_renew .* bit_vec + para_B40_ini .* bit_flip_vec
        ac[:,2] += bit_vec
        return para_B20_ini, para_B40_return, para_B44_ini, ac
    elseif para_renew_number == 3
        para_B44_return = para_B44_renew .* bit_vec + para_B44_ini .* bit_flip_vec
        ac[:,3] += bit_vec
        return para_B20_ini, para_B40_ini, para_B44_return, ac
    end
    
end


#レプリカ交換
function exchange_spc_burn(para_B20,para_B40,para_B44)
    #逆温度の差
    beta_dif = beta_list[2:end] - beta_list[1:end-1]

    # 誤差関数
    error_list = zeros(L)
    @threads for l in 1:L
        error_list[l] = error_spc(para_B20[l],para_B40[l],para_B44[l])
    end

    index_list = collect(1:L)

    for replica in 1:L-1
        #交換確率
        prob = exp(
            n * b_spc * beta_dif[replica] * (error_list[index_list[replica+1]]-error_list[index_list[replica]])
        )

        #ビット
        bit = rand() < prob # 1で交換、0で非交換

        if bit == 1 #交換
            tmp_num = index_list[replica]
            index_list[replica] = index_list[replica + 1]
            index_list[replica + 1] = tmp_num
        end

    end

    return para_B20[index_list],para_B40[index_list],para_B44[index_list]
end


#レプリカ交換
function exchange_spc(para_B20,para_B40,para_B44,ex_rate)
    #逆温度の差
    beta_dif = beta_list[2:end] - beta_list[1:end-1]

    # 誤差関数
    error_list = zeros(L)
    @threads for l in 1:L
        error_list[l] = error_spc(para_B20[l],para_B40[l],para_B44[l])
    end

    index_list = collect(1:L)

    for replica in 1:L-1
        #交換確率
        prob = exp(
            n * b_spc * beta_dif[replica] * (error_list[index_list[replica+1]]-error_list[index_list[replica]])
        )

        #ビット
        bit = rand() < prob # 1で交換、0で非交換

        if bit == 1 #交換
            tmp_num = index_list[replica]
            index_list[replica] = index_list[replica + 1]
            index_list[replica + 1] = tmp_num
            ex_rate[replica] += bit
        end

    end

    return para_B20[index_list],para_B40[index_list],para_B44[index_list], ex_rate
end


#事前分布による初期パラメータ決定
function initial_para_make()
    #パラメータ候補値
    para_num = collect(0.1:0.01:0.5)
    #確率
    para_prob = prior.(para_num)
    return sample(para_num, ProbabilityWeights(para_prob),L)
end


function exmc(iter_num_burn, iter_num)
    #初期パラメータ設定
    para_B20 = initial_para_make() 
    para_B40 = initial_para_make()
    para_B44 = initial_para_make() 

    #採択率・交換率
    ac = zeros(L,3)
    ex_rate = zeros(L)

    #パラメータ
    B20_save = zeros(iter_num, L)
    B40_save = zeros(iter_num, L)
    B44_save = zeros(iter_num, L)

    
    # burn in
    # print
    print_num_burn = Int16(iter_num_burn / 10)
    for iter in 1:iter_num_burn
        para_B20,para_B40,para_B44 = metropolis_spc_burn(para_B20,para_B40,para_B44)
        para_B20,para_B40,para_B44 = metropolis_spc_burn(para_B20,para_B40,para_B44)
        para_B20,para_B40,para_B44 = metropolis_spc_burn(para_B20,para_B40,para_B44)
        
        if iter % 5 == 0
            para_B20,para_B40,para_B44 = exchange_spc_burn(para_B20,para_B40,para_B44)
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
        para_B20,para_B40,para_B44, ac = metropolis_spc(para_B20,para_B40,para_B44, ac)
        para_B20,para_B40,para_B44, ac = metropolis_spc(para_B20,para_B40,para_B44, ac)
        para_B20,para_B40,para_B44, ac = metropolis_spc(para_B20,para_B40,para_B44, ac)
        B20_save[iter,:] = para_B20
        B40_save[iter,:] = para_B40
        B44_save[iter,:] = para_B44

        
        if iter % 5 == 0
            para_B20,para_B40,para_B44, ex_rate = exchange_spc(para_B20,para_B40,para_B44, ex_rate)
        end
        

        if iter % print_num == 0
            iter_perent = iter / print_num * 10
            println("finish $iter_perent percent")
        end
    end

    ac /= iter_num
    ex_rate /= (iter_num / 5)

    return B20_save, B40_save, B44_save, ac, ex_rate
end

para_B20, para_B40, para_B44, ac, ex_rate = exmc(iter_num_burn, iter_num)

# rate保存
ac_ex_path = path * "ac_ex.h5"
h5open(ac_ex_path, "w") do file
    write(file, "ac", ac)
    write(file, "ex", ex_rate)
end


# パラメータ保存
para_path = path * "para.h5"
h5open(para_path, "w") do file
    write(file, "B20", para_B20)
    write(file, "B40", para_B40)
    write(file, "B44", para_B44)
end