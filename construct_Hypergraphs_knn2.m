function [L, AA, W_norm]= construct_Hypergraphs_knn2(H,model)
% using knn with each node has equal knns
% H = H'; 
De = diag(sum(H,1)) + 0.0001*eye(size(H,2));
Dn = diag(sum(H,2));
w = diag(ones(1,size(De,2))); % hyperedge weight equals to 1 for each one
% w = assign_weight_for_hedge(data, H);

if strcmp(model,'Zhou')
    DeZhou = De;
    AA = H*DeZhou^(-1)*w*H';
    % W =  H*w*H' - Dn;
elseif strcmp(model,'Rod')
    AA = H*w*H' - Dn;
    Dn = diag(sum(AA));
elseif strcmp(model,'Saito')
    De = De - diag(ones(1,size(De,2)));
    A = H*De^(-1)*w*H';
    AA = A-diag(diag(A));
else
    error('You have to choose mode from either Zhou, Rod, or Saito')
end
L = Dn^(-1/2)*(Dn - AA)*Dn^(-1/2);
W_norm = Dn^(-1/2)*AA*Dn^(-1/2);
end
