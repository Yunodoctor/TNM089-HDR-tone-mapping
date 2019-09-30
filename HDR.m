clc
clear all;
image6 = imread('2.jpg');
image5 = imread('2.jpg');
image4 = imread('1.jpg');
image3 = imread('1.jpg');
image2 = imread('0.jpg');
image1 = imread('0.jpg');

im1 = double(image1);
im2 = double(image2);
im3 = double(image3);
im4 = double(image4);
im5 = double(image5);
im6 = double(image6);

s = 682*1023; 
% Add together each image, from each channel to one long row of the images
R = [reshape(im1(:,:,1),[s,1]),reshape(im2(:,:,1),[s,1]),reshape(im3(:,:,1),[s,1]),reshape(im4(:,:,1),[s,1]),reshape(im5(:,:,1),[s,1]),reshape(im6(:,:,1),[s,1])];
G = [reshape(im1(:,:,2),[s,1]),reshape(im2(:,:,2),[s,1]),reshape(im3(:,:,2),[s,1]),reshape(im4(:,:,2),[s,1]),reshape(im5(:,:,2),[s,1]),reshape(im6(:,:,2),[s,1])];
B = [reshape(im1(:,:,3),[s,1]),reshape(im2(:,:,3),[s,1]),reshape(im3(:,:,3),[s,1]),reshape(im4(:,:,3),[s,1]),reshape(im5(:,:,3),[s,1]),reshape(im6(:,:,3),[s,1])];

%lambda in [1,5]
l = 100;

%take 1000 random samples from R,G,B
% x = round(1000*rand);
% x = 1000;
% sample = randperm(size(R,1),x);

%take evenly distributed pixels
sample_even = (1:300:size(R,1));
x = size(sample_even,2);
R_sub = zeros(x,6);
G_sub = zeros(x,6);
B_sub = zeros(x,6);

% for j=1:x
%     R_sub(j,:) = R(sample(j),:);
%     G_sub(j,:) = G(sample(j),:);
%     B_sub(j,:) = B(sample(j),:);
% end

for j=1:x
    R_sub(j,:) = R(sample_even(j),:);
    G_sub(j,:) = G(sample_even(j),:);
    B_sub(j,:) = B(sample_even(j),:);
end

%Ax=B, get the B
%b is the exposures for each image. Use the info on the image later pls.
b  = [1/60;1/30;1/15;1/8;1/4;1/2];
b = log(b);
b1 = zeros(x*6,6);
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

eR = zeros(s,6);
eG = zeros(s,6);
eB = zeros(s,6);

for m = 1:256
    for n = 1:6
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
for n1 = 1:6
    sum_R = sum_R + eR(:,n1);
    sum_G = sum_G + eG(:,n1);
    sum_B = sum_B + eB(:,n1);
end

% takes the median for each pixel of each image in each channel.
radiance_mapR = sum_R/6;
radiance_mapG = sum_G/6;
radiance_mapB = sum_B/6;

radiance_mapR = reshape(radiance_mapR,[682,1023]);
radiance_mapG = reshape(radiance_mapG,[682,1023]);
radiance_mapB = reshape(radiance_mapB,[682,1023]);

EnormR = zeros(682,1023);
EnormG = zeros(682,1023);
EnormB = zeros(682,1023);

minE = min([min(radiance_mapR(:)),min(radiance_mapG(:)),min(radiance_mapB(:))]);
maxE = max([max(radiance_mapR(:)),max(radiance_mapG(:)),max(radiance_mapB(:))]);

%normalization to be able to gamma (Jacobs killgissning)
for k = 1:682 
   for l = 1:1023
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

Enorm3 = cat(3,EnormR,EnormG,EnormB);
M = tonemap(Enorm3);

Rnew = zeros(682,1023);
Gnew = zeros(682,1023);
Bnew = zeros(682,1023);

% Adding the tonemap to the final Image
for k3 = 1:682
    for k4 = 1:1023
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
imshow (newI) % M = Ltone./L1;
title ('Reinhard')


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




