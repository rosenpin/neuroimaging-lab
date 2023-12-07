% read the header
% go through all files that end with nii and read them
files = dir('c*.nii');
for i=1:length(files)
    fileName = files(i).name;
    V=spm_vol(fileName);
    % read the volume
    Y=spm_read_vols(V);
    % check dimensions
    size(Y)
    % display a single slice (you should choose one from the middle)
    figure;
    imagesc(Y(:,:,55));
end