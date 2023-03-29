close all;
clear all;



% read in data
data = load('data/train_grass.txt');
dim = size(data);
new_data = zeros(dim(1,1),dim(1,2)*2);
new_data(1:dim(1,1),1:dim(1,2)) = data;

% modification on data
for i = 1 :dim(1,2)
    trans_temp = data(:,i);
    trans_temp = reshape(trans_temp,[8,8]);
    trans_temp = trans_temp+normrnd(0,0.2);
    trans_temp = fliplr(trans_temp);
    trans_temp = reshape(trans_temp,[64,1]);
    trans_temp = trans_temp(randperm(length(trans_temp)));
    new_data(:,dim(1,2)+i) = trans_temp;
end

fp=fopen('data/train_grass_flipnoise.txt','wt');
% fp=fopen('data/train_grass_noise.txt','wt');
% write into new dataset
dim = size(new_data);
for i = 1 : dim(1,1)
    for j = 1:dim(1,2)
        if j == dim(1,2)
            fprintf(fp,'%f\n',new_data(i,j));
        else
            fprintf(fp,'%f,',new_data(i,j));
        end
    end
end
fclose(fp);