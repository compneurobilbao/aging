function generate_mod(nMod)

    cd /beegfs/home/aerramuzpe/CLUSTER
    load('FCPIL_SCPIL.mat')
    cd /beegfs/home/aerramuzpe/CLUSTER/mod_20_2514

    %% FC_mod SC_mod for a given nMod
    modules_indx=modules_20_60(nMod,1:nMod);
    'FC & SC descriptors calculations'
    %     tic
    FC_Mod=single(zeros(nMod,nMod,length(subjIdxList)));
    SC_Mod=single(zeros(nMod,nMod,length(subjIdxList)));

    for i=1:nMod
	for j=i:nMod
	    A=FCpil(modules_indx{i},modules_indx{j},:);
	    FC_Mod(i,j,:)=sum(sum(A,1),2)/(length(modules_indx{i})*length(modules_indx{j}));
	    FC_Mod(j,i,:)=FC_Mod(i,j,:);
	    B=double(SCpil(modules_indx{i},modules_indx{j},:));
	    SC_Mod(i,j,:)=sum(sum(B,1),2)/(length(modules_indx{i})*length(modules_indx{j}));
	    SC_Mod(j,i,:)=SC_Mod(i,j,:);
	end
    end
    %         toc
    clear A B
    save (['mod_' num2str(nMod)],'FC_Mod','SC_Mod')
end
