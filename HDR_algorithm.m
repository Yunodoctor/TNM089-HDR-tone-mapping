% HDR algorithm
% Made by Emilie Ho, Viktor Sandberg and Jacob Nyman
% The algorithm originates from: Paul Debevec's SIGGRAPH'97 paper "Recovering High Dynamic Range
% Radiance Maps from Photographs" 
clear all
clc

take_even_sample = 1;

% Read in pictures from folder
imageFolder = 'C:\Users\jacob\Skola\År 5\TNM089\TNM089_HDR_tone_mapping\images\ClockTower';
if ~isdir(imageFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', imageFolder);
  uiwait(warndlg(errorMessage));
  return;
end

filePattern = fullfile(imageFolder, '*.jpg');
jpegFiles = dir(filePattern);
for i = 1:length(jpegFiles)
  baseFileName = strcat('image (',num2str(i),').jpg');
  fprintf(1, 'Reading images %s\n', baseFileName);
  fullFileName = fullfile(imageFolder, baseFileName);
  imageInfo{i} = fullFileName;
  image{i} = imread(fullFileName);
  imgCropped{i} = imcrop(image{i}, [2000 2000 900 700]);
  im{i} = double(imgCropped{i});
end

imgSizeOneDim = size(im{1}, 1) * size(im{1}, 2);
imgDims = im{1};
% Adding each channel of the images to an array.
for nrImages = 1:size(image,2)
    R(:,nrImages) = reshape(im{nrImages}(:,:,1), [imgSizeOneDim, 1]);
    G(:,nrImages) = reshape(im{nrImages}(:,:,2), [imgSizeOneDim, 1]);
    B(:,nrImages) = reshape(im{nrImages}(:,:,3), [imgSizeOneDim, 1]);
end

%lambda in [1,5]
l = 100;
if take_even_sample == 1
    %take evenly distributed pixels
    sample_even = (1:300:size(R,1));
    x = size(sample_even,2);
    for j=1:x
        R_sampled(j,:) = R(sample_even(j),:);
        G_sampled(j,:) = G(sample_even(j),:);
        B_sampled(j,:) = B(sample_even(j),:);
    end
else
    %take 1000 random samples from R,G,B
     x = 1000;
     sample_rand = randperm(size(R,1),x);
     for j=1:x
         R_sampled(j,:) = R(sample_rand(j),:);
         G_sampled(j,:) = G(sample_rand(j),:);
         B_sampled(j,:) = B(sample_rand(j),:);
     end
end
%%
for exposures = 1:nrImages
    info = imfinfo(imageInfo{exposures});
    expoTimeArray(exposures) = info.DigitalCamera.ExposureTime;
end

logExpoTimeArray = log(expoTimeArray);

% fix exposures to be of nrOfImages
% B is the log delta t, or log shutter speed, for image j

for k=1:size(logExpoTimeArray,1)
    B(:,k) = logExpoTimeArray(k);
end



%%
% Calculating the weights for the HDR algorithm
W = zeros(256,1);
for i=1:256
    W(i) = weight(i);
end

% Using the orignal HDR algorithm from "97 to get the responsefunction
% g_RGB(z) is the response curve 
% lE_RGB(i) is the log exposure(irradiance)

[g_R,lE_R] = RadianceMapAlgorithm(R_sampled,B,l,W);
[g_G,lE_G] = RadianceMapAlgorithm(G_sampled,B,l,W);
[g_B,lE_B] = RadianceMapAlgorithm(B_sampled,B,l,W);

%% Making the radiance map

sum_R = 0;
sum_G = 0;
sum_B = 0;

for m = 1:256
    for n = 1:nrImages
        %Finding all the color values in the channel (0-255) and return the
        %index
        indices_R = find(R(:,n) == m-1);
        indices_G = find(G(:,n) == m-1);
        indices_B = find(B(:,n) == m-1);
        
        % Responsefunction minus exposure in that
        % equation 5 in the paper. "Radiance value"
        Exp_R(indices_R,n) = g_R(m)-B(n);
        Exp_G(indices_G,n) = g_G(m)-B(n);
        Exp_B(indices_B,n) = g_B(m)-B(n);
    end
end

for c = 1:nrImages
% Add the color channels for each image.
    sum_R = sum_R + Exp_R(:,c);
    sum_G = sum_G + Exp_G(:,c);
    sum_B = sum_B + Exp_B(:,c);  
end

% takes the mean for each pixel of each image in each channel and reshapes
% them from a single row to an image again
radiance_mapR = reshape(sum_R/nrImages,[size(imgDims,1),size(imgDims,2)]);
radiance_mapG = reshape(sum_G/nrImages,[size(imgDims,1),size(imgDims,2)]);
radiance_mapB = reshape(sum_B/nrImages,[size(imgDims,1),size(imgDims,2)]);

% Calculate min and max radiance value
minE = min([min(radiance_mapR(:)),min(radiance_mapG(:)),min(radiance_mapB(:))]);
maxE = max([max(radiance_mapR(:)),max(radiance_mapG(:)),max(radiance_mapB(:))]);

%Normalising based on the max and min radiance to [0,1]
for i = 1:size(imgDims,1) 
   for j = 1:size(imgDims,2)
         E_normR(i,j) = (radiance_mapR(i,j)-minE)/(maxE-minE);
         E_normG(i,j) = (radiance_mapG(i,j)-minE)/(maxE-minE);
         E_normB(i,j) = (radiance_mapB(i,j)-minE)/(maxE-minE);
   end
end

% Gamma correction: in book: eq: 10.9
% gamma[0,1] = regulates the contrast lower value lower contrast
% alpha[0,1] < 1 decresing the exposure
gamma = 0.5;
A = 0.75;
gammaCorrectedImage(:,:,1) = A*E_normR.^gamma;
gammaCorrectedImage(:,:,2) = A*E_normG.^gamma;
gammaCorrectedImage(:,:,3) = A*E_normB.^gamma;

ToneMapping = ReinhardDevlinTMO(gammaCorrectedImage);

% Applying the tonemapping to the normalized image
for i = 1:size(imgDims, 1)
    for j = 1:size(imgDims, 2)
        outputImg(i,j,1) = ToneMapping(i,j)*gammaCorrectedImage(i,j,1);
        outputImg(i,j,2) = ToneMapping(i,j)*gammaCorrectedImage(i,j,2);
        outputImg(i,j,3) = ToneMapping(i,j)*gammaCorrectedImage(i,j,3);
    end
end

imshow(outputImg)
