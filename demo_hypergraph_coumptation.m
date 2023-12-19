%% demo for constructing hypergraph
% streoseq data
data_coo = importdata('Stero_seq\Dataset1_LiuLongQi_MouseOlfactoryBulb_co.csv');
W_euc = dist2(data_coo, data_coo);
k_nn = 20; 
model = 'Zhou';
H = zeros(size(W_euc));
for i=1:size(W_euc,1)
    ll = W_euc(i,:);
    [~, index_i] = sort(ll);
    k_ii = index_i(2:k_nn + 1);
    H(i,k_ii) = 1;
    clear index_i k_ii ll
end
H = H + eye(size(H));
[L, AA, W_norm]= construct_Hypergraphs_knn2(H, model);
% csvwrite('data\Stero_seq\Steroseq_adj_norm.csv', W_norm);
clear all