function [num_mods_struct,num_mods_func,splitted_SC,splitted_FC] = internal()
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
clear all; clc
currentFolder = pwd;
EDGEcn=cell(1,2514);
EDGEpn=cell(1,2514);

load('responses.mat') %Select the appropiate one
gainSubj =  sigcsf_ab42x; % or whatever
for nMod=21:1000
    nMod
    data = load(strcat(currentFolder,'/mod/mod_',num2str(nMod)));
    SC_Mod_pil=data.SC_Mod;

    EDGEcn{nMod}=(zeros(nMod,1));
    EDGEpn{nMod}=(zeros(nMod,1));

    for i=1:nMod
        %total degree - i (out-degree)
        SC_Mod_pil_out_degree(i,:) = SC_Mod_pil(i,i,:);
    end

    [EDGEcn{nMod},EDGEpn{nMod}]=corr(SC_Mod_pil_out_degree',gainSubj);
end
save -v7 data_internal_sigcsf_ab42x_21_1000.mat EDGEcn EDGEpn


clear all; clc
currentFolder = pwd;
EDGEcn=cell(1,2514);
EDGEpn=cell(1,2514);

load('responses.mat') %Select the appropiate one
gainSubj =  sigcsf_ptau;
for nMod=21:1000
    nMod

    data = load(strcat(currentFolder,'/mod/mod_',num2str(nMod)));
    SC_Mod_pil=data.SC_Mod;

    EDGEcn{nMod}=(zeros(nMod,1));
    EDGEpn{nMod}=(zeros(nMod,1));

    for i=1:nMod
        %total degree - i (out-degree)
        SC_Mod_pil_out_degree(i,:) = SC_Mod_pil(i,i,:);
    end

    [EDGEcn{nMod},EDGEpn{nMod}]=corr(SC_Mod_pil_out_degree',gainSubj);
end
save -v7 data_internal_sigcsf_ptau_21_1000.mat EDGEcn EDGEpn


clear all; clc
currentFolder = pwd;
EDGEcn=cell(1,2514);
EDGEpn=cell(1,2514);

load('responses.mat') %Select the appropiate one
gainSubj =  sigcsf_tau;
for nMod=21:1000
    nMod
    data = load(strcat(currentFolder,'/mod/mod_',num2str(nMod)));

    SC_Mod_pil=data.SC_Mod;

    EDGEcn{nMod}=(zeros(nMod,1));
    EDGEpn{nMod}=(zeros(nMod,1));

    for i=1:nMod
        %total degree - i (out-degree)
        SC_Mod_pil_out_degree(i,:) = SC_Mod_pil(i,i,:);
    end

    [EDGEcn{nMod},EDGEpn{nMod}]=corr(SC_Mod_pil_out_degree',gainSubj);
end
save -v7 data_internal_sigcsf_tau_21_1000.mat EDGEcn EDGEpn

end










