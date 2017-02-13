function [num_mods_struct,num_mods_func,splitted_SC,splitted_FC] = internal()
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

clear all; clc

pValueTh=0.05;
load('/beegfs/home/aerramuzpe/CLUSTER/partition_20_2514.mat')
%load('/beegfs/home/aerramuzpe/CLUSTER/aalInfoMatch.mat')


modAll_nMod=cell(2514);
initiate
cd /beegfs/home/aerramuzpe/CLUSTER

nModIn=20;
for nMod=21:1000
    nMod
    newMod=[];
    oldMod=[];

    modules_indx=modules_20_60(nMod,1:nMod); %modules_20_60{}
    modules_indx20=modules_20_60(20,1:20);
    modules_indxIn=modules_20_60(nMod-1,1:nMod-1);
    for i=1:length(modules_indx)
        for j=1:length(modules_indx20)
            if length(modules_indx{i})<length(modules_indx20{j}) & length(intersect(modules_indx{i},modules_indx20{j}))==length(modules_indx{i})
                newMod=[newMod i];
                oldMod=[oldMod j];
            end
        end
    end
    modules_indxPrev=modules_indx;
    newPrevMod{nMod}=[newMod; oldMod];

    data = load(strcat('/beegfs/home/aerramuzpe/CLUSTER/mod_20_2514/mod_',num2str(nMod)));

    FC_Mod_pil=data.FC_Mod;
    SC_Mod_pil=data.SC_Mod;

    %% SC and FC edge correlation with age
    EDGEcn{nMod}=(zeros(nMod,1));
    EDGEpn{nMod}=(zeros(nMod,1));
    EDGEcnRS{nMod}=(zeros(nMod,1));
    EDGEpnRS{nMod}=(zeros(nMod,1));


    for i=1:nMod
        %total degree - i (out-degree)
        SC_Mod_pil_out_degree(i,:) = SC_Mod_pil(i,i,:);
        FC_Mod_pil_out_degree(i,:) = FC_Mod_pil(i,i,:);

        [r,p]=corrcoef(SC_Mod_pil_out_degree(i,:)',gainSubj);
        EDGEcn{nMod}(i)=r;
        EDGEpn{nMod}(i)=p;
        [r,p]=corrcoef(FC_Mod_pil_out_degree(i,:)',gainSubj); %ageSubj
        EDGEcnRS{nMod}(i)=r;
        EDGEpnRS{nMod}(i)=p;

    end

    %% STRUCTURAL
    % significant SC edges in new partition - written also as a string, edgeNew
    edge_SC{nMod}=find((EDGEpn{nMod}<(pValueTh/nMod)) & (EDGEpn{nMod}>eps)); %this is the idx, not the module
    %[u v]=ind2sub([nMod nMod],signD);
    %edge_SC{nMod}=sort([u v],2);
    num_mods_struct(nMod)=length(edge_SC{nMod});
    disp(['SC sig: ' num2str(num_mods_struct(nMod))]);
    edgeNew_SC={''};
    for i=1:size(edge_SC{nMod},1)
        edgeNew_SC=[edgeNew_SC num2str(edge_SC{nMod}(i,:))];
    end
    % identification in old partition of edges involving the module
    % splitted in two
    
    %% test with previous edge_SC{nMod-1}
    if nMod>(nModIn)
        prevMod=newPrevMod{nMod}(2,1);
        if find(edge_SC{nMod-1}==newPrevMod{nMod}(2,1)) %looking for the splitted one. 
            oldEdgesInSC=ismember(num2str(1),edgeNew_SC) + ismember(num2str(2),edgeNew_SC);
            disp(['SC Modules= ' num2str(nMod) ' ;from the 2 splited, in this one: ' num2str(length(oldEdgesInSC))]);
            splitted_SC(nMod)=1;
	    significant_SC(nMod)=num2str(length(oldEdgesInSC));
        else
            disp(['SC Modules= ' num2str(nMod) '; The splitted module was not significant'])
            splitted_SC(nMod)=NaN;
            significant_SC(nMod)=NaN;
        end
    end

    
    %% FUNCTIONAL  
    % significant FC edges in new partition - written also as a string, edgeNew
    edge_FC{nMod}=find((EDGEpnRS{nMod}<(pValueTh/nMod)) & (EDGEpnRS{nMod}>eps)); %this is the idx, not the module
    %[u v]=ind2sub([nMod nMod],signD);
    %edge_FC{nMod}=sort([u v],2);
    num_mods_func(nMod)=length(edge_FC{nMod});
    disp(['FC sig: ' num2str(num_mods_func(nMod))]);
    edgeNew_FC={''};
    for i=1:size(edge_FC{nMod},1)
        edgeNew_FC=[edgeNew_FC num2str(edge_FC{nMod}(i,:))];
    end
    % identification in old partition of edges involving the module
    % splitted in two
    
    %% test with previous edge_SC{nMod-1}
    if nMod>(nModIn)
        prevMod=newPrevMod{nMod}(2,1);
        if find(edge_FC{nMod-1}==newPrevMod{nMod}(2,1)) %looking for the splitted one. 
            oldEdgesInFC=ismember(num2str(1),edgeNew_FC) + ismember(num2str(2),edgeNew_FC);
            disp(['FC Modules= ' num2str(nMod) ' ;from the 2 splited, in this one: ' num2str(length(oldEdgesInFC))]);
            splitted_FC(nMod)=1;
            significant_FC(nMod)=num2str(length(oldEdgesInFC));
        else
            disp(['FC Modules= ' num2str(nMod) '; The splitted module was not significant'])
            splitted_FC(nMod)=NaN;
            significant_FC(nMod)=NaN;
        end
    end
    
end
save -mat7-binary data_internal_21_1000.mat newPrevMod EDGEcn EDGEpn EDGEcnRS EDGEpnRS edge_FC edge_SC ...
num_mods_struct num_mods_func splitted_SC splitted_FC
end


