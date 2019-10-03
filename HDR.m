clc
clear all;

% Read in pictures from folder
cd images
cd Seine
folderPath = pwd;
cd ..
cd ..

filePattern = dir(fullfile(folderPath, '*.jpg'));

for i = 1:length(filePattern)
    fileName = filePattern(i).name;
    fullName = fullfile(folderPath, fileName);
    fprintf(1, 'Reading images %s\n', fullName);
    image{i} = imread(fullName);
    im{i} = double(image{i});
end
%s = 682*1023; 
s = size(im{1}, 1) * size(im{i}, 2);
image1 = im{1};

for nrOfImages = 1:size(image,2)
    reshapedImageR{nrOfImages} = reshape(im{nrOfImages}(:,:,1),[s,1]);
    reshapedImageG{nrOfImages} = reshape(im{nrOfImages}(:,:,2),[s,1]);
    reshapedImageB{nrOfImages} = reshape(im{nrOfImages}(:,:,3),[s,1]); 
end

% Add together each image, from each channel to one long row of the images
Rcell = [reshapedImageR(:,1:nrOfImages)];
Gcell = [reshapedImageG(:,1:nrOfImages)];
Bcell = [reshapedImageB(:,1:nrOfImages)];
R = cell2mat(Rcell);
G = cell2mat(Gcell);
B = cell2mat(Bcell);

%lambda in [1,5]
l = 100;

%take 1000 random samples from R,G,B
% x = round(1000*rand);
% x = 1000;
% sample = randperm(size(R,1),x);

%take evenly distributed pixels
sample_even = (1:300:size(R,1));
x = size(sample_even,2);
R_sub = zeros(x,nrOfImages);
G_sub = zeros(x,nrOfImages);
B_sub = zeros(x,nrOfImages);

% for j=1:x
%     R_sub(j,:) = R(sample(j),:);
%     G_sub(j,:) = G(sample(j),:);
%     B_sub(j,:) = B(sample(j),:);
% end

for k=1:x
    R_sub(k,:) = R(sample_even(k),:);
    G_sub(k,:) = G(sample_even(k),:);
    B_sub(k,:) = B(sample_even(k),:);
end

%Ax=B, get the B
%b is the exposures for each image. Use the info on the image later pls.
b  = [1/60;1/30;1/15;1/8;1/4;1/2];
%fix exposures to be of nrOfImages

b = log(b);
b1 = zeros(x*nrOfImages,nrOfImages);
for j=1:size(b,1)
    b1(:,j) = b(j);
end

% b1 The log exposure for each image

% Calculate the weights 
W = zeros(256,1);
for i=1:256
    W(i) = weight(i);
end

% Use the HDR function to get the responsfunctions 
% g_RGB(z) is the log exposure corresponding to pixel value z
% lE_RGB(i) is the log film irradiance at pixel location i

[g_R,lE_R] = gsolve(R_sub,b1,l,W);
[g_G,lE_G] = gsolve(G_sub,b1,l,W);
[g_B,lE_B] = gsolve(B_sub,b1,l,W);


%X_R = zeros(x,6);
%for k=1:6
%X_R(:,k) = lE_R + b1(1:x,k);
%end

%% Radiance map
sum_R = 0;
sum_G = 0;
sum_B = 0;

eR = zeros(s,nrOfImages);
eG = zeros(s,nrOfImages);
eB = zeros(s,nrOfImages);

for m = 1:256
    for n = 1:nrOfImages
        %Finding all the color values in the channel (0-255) and return the
        %index
        indices_R = find(R(:,n) == m-1);
        indices_G = find(G(:,n) == m-1);
        indices_B = find(B(:,n) == m-1);
        
        % Responsfunction minus exponering i det indexet
        % equation 5 in the paper. "Radiance value"
        eR(indices_R,n) = g_R(m)-b(n);
        eG(indices_G,n) = g_G(m)-b(n);
        eB(indices_B,n) = g_B(m)-b(n);
    end
end

% Add all color channels together for each pixel
for n1 = 1:nrOfImages
    sum_R = sum_R + eR(:,n1);
    sum_G = sum_G + eG(:,n1);
    sum_B = sum_B + eB(:,n1);
end

% takes the median for each pixel of each image in each channel.
radiance_mapR = sum_R/nrOfImages;
radiance_mapG = sum_G/nrOfImages;
radiance_mapB = sum_B/nrOfImages;

radiance_mapR = reshape(radiance_mapR,[size(image1,1),size(image1,2)]);
radiance_mapG = reshape(radiance_mapG,[size(image1,1),size(image1,2)]);
radiance_mapB = reshape(radiance_mapB,[size(image1,1),size(image1,2)]);

EnormR = zeros(size(image1,1),size(image1,2));
EnormG = zeros(size(image1,1),size(image1,2));
EnormB = zeros(size(image1,1),size(image1,2));

minE = min([min(radiance_mapR(:)),min(radiance_mapG(:)),min(radiance_mapB(:))]);
maxE = max([max(radiance_mapR(:)),max(radiance_mapG(:)),max(radiance_mapB(:))]);

%normalization to be able to gamma (Jacobs killgissning)
for k = 1:size(image1,1) 
   for l = 1:size(image1,2)
         EnormR(k,l) = (radiance_mapR(k,l)-minE)/(maxE-minE);
         EnormG(k,l) = (radiance_mapG(k,l)-minE)/(maxE-minE);
         EnormB(k,l) = (radiance_mapB(k,l)-minE)/(maxE-minE);
   end
end

% Gamma correction: in book: eq: 10.9
% gamma = regulates the contrast lower value lower constra
% alpha < 1 decresing the exposure

gamma = 0.5;
A = 0.75;

EgammaR = A*EnormR.^gamma;
EgammaG = A*EnormG.^gamma;
EgammaB = A*EnormB.^gamma;

imageGamma = cat(3,EgammaR,EgammaG,EgammaB);
%% Tonemapping
Enorm3 = cat(3,EnormR,EnormG,EnormB);

%DurandTMO
%GammaTMO
%KimKautzConsistentTMO
%ReinhardTMO <- crazy
%SchlickTMO <- Inverted
%KuangTMO <- Mörka kontraster
M = ReinhardDevlinTMO(Enorm3);

Rnew = zeros(size(image1,1),size(image1,2));
Gnew = zeros(size(image1,1),size(image1,2));
Bnew = zeros(size(image1,1),size(image1,2));

% Adding the tonemap to the final Image
for k3 = 1:size(image1,1)
    for k4 = 1:size(image1,2)
        Rnew(k3,k4) = M(k3,k4)*EnormR(k3,k4);
        Gnew(k3,k4) = M(k3,k4)*EnormG(k3,k4);
        Bnew(k3,k4) = M(k3,k4)*EnormB(k3,k4);
    end
end

newI = cat(3,Rnew,Gnew,Bnew);

% Gamma Image
figure
imshow(imageGamma)
title('Gamma')

% Filter 'Reinhard*
figure
imshow (colormap(newI,'hot')) % M = Ltone./L1;
title ('TONE MAPPED')


%% Plots
subplot(2,2,1)
plot(g_R,(0:255),'r')
title('Response curve - Red')
hold on

subplot(2,2,2)
plot(g_G,(0:255),'g')
title('Response curve - Green')
hold on

subplot(2,2,3)
plot(g_B,(0:255),'b')
title('Response curve - Blue')
hold on

subplot(2,2,4)
plot(g_R,(0:255),'r')
hold on 
plot(g_G,(0:255),'g')
hold on
plot(g_B,(0:255),'b')

% plot(X_R(:,1),R_sub(:,1),'.')
% hold on 
% plot(X_R(:,2),R_sub(:,2),'.')
% hold off




