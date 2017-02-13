%% fix gender, age range, if using DTI, rs, PCA
ageMin=0;
ageMax=120;
gender=''; % F M or empty
pcaAn='no';
DTI='yes';
rs='yes';
brainPart='SFM'; % rsn20 % rsn % SFM20

%% load data sets
load('partition_20_2514.mat')

dataSet={'/beegfs/home/aerramuzpe/CLUSTER/life_span_paolo','/beegfs/home/aerramuzpe/CLUSTER/life_span2'};
ageSubj=[];
sexSubj=[];
subjID=cell(0);
for di=1:size(dataSet,2)
    cd (dataSet{di})
    allData=dir;
    allData(1:2)=[];
    if strcmp(dataSet{di},'/beegfs/home/aerramuzpe/CLUSTER/life_span_paolo')
        for i=1:length(allData)
            if allData(i).isdir
                %                 cd (allData(i).name)
                s=importdata([allData(i).name '/gender.txt']) ;
                a=importdata([allData(i).name '/age.txt']);
                ageSubj=[ageSubj a];
                sexSubj=[sexSubj s];
                subjID=[subjID num2str(i)];
            end
        end
    end
    if strcmp(dataSet{di},'/beegfs/home/aerramuzpe/CLUSTER/life_span2')
        load ('partecipantsInfo_v2.mat')
        ageSubj=[ageSubj (cell2mat(youngAge))'];
        sexSubj=[sexSubj ((youngSex))'];
        subjID=[subjID ((youngID))'];
        ageSubj=[ageSubj (cell2mat(oldAge))'];
        sexSubj=[sexSubj ((oldSex))'];
        subjID=[subjID ((oldID))'];
    end
end

%% identify out of all subjects the good one matching sex and age
subjIdxListAge=find(ageSubj>=ageMin & ageSubj<=ageMax);
subjIdxListSex=1:length(ageSubj);
if ~isempty(gender)
    subjIdxListSexH=strcmpi(sexSubj,gender);
    subjIdxListSex=subjIdxListSex(subjIdxListSexH);
end
subjIdxList=intersect(subjIdxListAge,subjIdxListSex);
[v,p]=intersect(subjIdxList,[134 139 140 152 154]);
subjIdxList(p)=[];
gainSubj=ageSubj(subjIdxList)';

%% load all FC and SC matrixes and time series
FC=cell(1,length(subjIdxList));
SC=cell(1,length(subjIdxList));
SC_bin=cell(1,length(subjIdxList));
cd('/beegfs/home/aerramuzpe/CLUSTER/life_span_paolo')
FCpil=zeros(2514,2514,length(FC));
SCpil=zeros(2514,2514,length(SC));
allData=dir;
allData(1:2)=[];
conf=0;
coP=0;
for subjIdx=subjIdxList(subjIdxList<=118)
    conf=conf+1
    if strcmp(rs,'yes')
        load ([allData(subjIdx).name '/time_series.mat'])
        f=corrcoef(time_series);
        nIsnanPt=length(find(isnan(f(:))));
        fOk=f(~isnan(f(:)));
        rp=randperm(length(fOk));
        f(isnan(f(:)))=fOk(rp(1:nIsnanPt));
        coP=coP+length(find(isnan(f(:))))
        tmSer{conf}=time_series;
        FCpil(:,:,conf)=sparse(f);
        nFrSubj(conf)=size(time_series,1);
        %         tmSer{conf}=[];
        if conf==1;
            fc_aver=FCpil(:,:,conf);
        else
            fc_aver=fc_aver+FCpil(:,:,conf);
        end
    end
    if strcmp(DTI,'yes')
        load ([allData(subjIdx).name '/fiber_num.mat'])
        SCpil(:,:,conf)=sparse(fiber_num);
        %         SC_bin{conf}=sparse(fiber_num>0);
        if conf==1;
            sc_aver=SCpil(:,:,conf);
        else
            sc_aver=sc_aver+SCpil(:,:,conf);
        end
    end
end
cd('/beegfs/home/aerramuzpe/CLUSTER/life_span2')
for subjIdx=subjIdxList(subjIdxList>118)
    conf=conf+1
    if strcmp(rs,'yes')
        load ([subjID{subjIdx} '/functional_networks.mat'])
        f=corrcoef(time_series);
        nIsnanPt=length(find(isnan(f(:))));
        fOk=f(~isnan(f(:)));
        rp=randperm(length(fOk));
        f(isnan(f(:)))=fOk(rp(1:nIsnanPt));
        coP=coP+length(find(isnan(f(:))))
        tmSer{conf}=time_series;
        FCpil(:,:,conf)=sparse(f);
        nFrSubj(conf)=size(time_series,1);
        %         tmSer{conf}=[];
        if conf==1;
            fc_aver=FCpil(:,:,conf);
        else
            fc_aver=fc_aver+FCpil(:,:,conf);
        end
    end
    if strcmp(DTI,'yes')
        load ([subjID{subjIdx} '/fiber_num.mat'])
        SCpil(:,:,conf)=sparse(fiber_num);
        %         SC_bin{conf}=sparse(fiber_num>0);
        if conf==1;
            sc_aver=SC{conf};
        else
            sc_aver=sc_aver+SCpil(:,:,conf);
        end
    end
end
fc_aver=fc_aver/length(subjIdxList);
sc_aver=sc_aver/length(subjIdxList);

cd('/beegfs/home/aerramuzpe/CLUSTER')

save -v7 FCPIL_SCPIL FCpil SCpil subjIdxList
