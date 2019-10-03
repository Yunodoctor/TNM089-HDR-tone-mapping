clc
clear all;

% Read in pictures from folder
folderPath = 'C:\Users\Emelie\Documents\TNM089_HDR_tone_mapping';
filePattern = dir(fullfile(folderPath, '*.jpg'));

for i = 1:length(filePattern)
    fileName = filePattern(i).name;
    fullName = fullfile(folderPath, fileName);
    fprintf(1, 'Reading images %s\n', fullName);
    image{i} = imread(fullName);
    im{i} = double(image{i});
end

s = 682*1023; 
for numberOfImg = 1:size(image,2)
    reshapedImageR{numberOfImg} = reshape(im{numberOfImg}(:,:,1),[s,1]);
    reshapedImageG{numberOfImg} = reshape(im{numberOfImg}(:,:,2),[s,1]);
    reshapedImageB{numberOfImg} = reshape(im{numberOfImg}(:,:,3),[s,1]); 
end

Rcell = [reshapedImageR(:,1:numberOfImg)];
Gcell = [reshapedImageG(:,1:numberOfImg)];
Bcell = [reshapedImageB(:,1:numberOfImg)];
R = cell2mat(Rcell);
G = cell2mat(Gcell);
B = cell2mat(Bcell);

%lambda in [1,5]
l = 100;

%take 1000 random samples from R,G,B
% x = round(1000*rand);
% x = 1000;
% sample = randperm(size(R,1),x);
 
%take 1000 evenly distributed pixels
sample_even = (1:300:size(R,1));
x = size(sample_even,2);
R_sub = zeros(x,numberOfImg);
G_sub = zeros(x,numberOfImg);
B_sub = zeros(x,numberOfImg);

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

W = zeros(256,1);
for i=1:256
    W(i) = weight(i);
end

%Ax=B, get the B
b  = [1/60;1/30;1/15;1/8;1/4;1/2];
b = log(b);
b1 = zeros(x*numberOfImg,numberOfImg);
for j=1:size(b,1)
    b1(:,j) = b(j);
end

[g_R,lE_R] = gsolve(R_sub,b1,l,W);
[g_G,lE_G] = gsolve(G_sub,b1,l,W);
[g_B,lE_B] = gsolve(B_sub,b1,l,W);

X_R = zeros(x,numberOfImg);
for k=1:numberOfImg
X_R(:,k) = lE_R + b1(1:x,k);
end

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

%%radiance map
sum_R = 0;
sum_G = 0;
sum_B = 0;

eR = zeros(s,numberOfImg);
eG = zeros(s,numberOfImg);
eB = zeros(s,numberOfImg);

for m = 1:256
    for n = 1:numberOfImg
        indices_R = find(R(:,n) == m-1);
        indices_G = find(G(:,n) == m-1);
        indices_B = find(B(:,n) == m-1);

        eR(indices_R,n) = g_R(m)-b(n);
        eG(indices_G,n) = g_G(m)-b(n);
        eB(indices_B,n) = g_B(m)-b(n);
    end
end

for n1 = 1:numberOfImg
    sum_R = sum_R + eR(:,n1);
    sum_G = sum_G + eG(:,n1);
    sum_B = sum_B + eB(:,n1);
end

W_R = W(R+1);
W_G = W(G+1);
W_B = W(B+1);

radiance_mapR = sum_R/numberOfImg;
radiance_mapG = sum_G/numberOfImg;
radiance_mapB = sum_B/numberOfImg;


radiance_mapR = reshape(radiance_mapR,[682,1023]);
radiance_mapG = reshape(radiance_mapG,[682,1023]);
radiance_mapB = reshape(radiance_mapB,[682,1023]);

delta = 7;
% radiance = cat(3,radiance_mapR,radiance_mapG,radiance_mapB);
% h = heatmap(radiance,'x','y','symmetric','false');

L_w = 0.2126*radiance_mapR+0.7152*radiance_mapG+0.0722*radiance_mapB;
 L_w_bar = exp(mean(log(L_w(:) + delta))); %%% delta is a small number
 L = (0.045/L_w_bar)*L_w;
 
%%% 0.18 is the middle-grey key value;
%%% You may set the value to 0.09, 0.36, 0.54, 0.72.
 %heatmap(flipud(L), 'colormap', 'jet', 'symmetric', 'false')

 EnormR = zeros(682,1023);
 EnormG = zeros(682,1023);
 EnormB = zeros(682,1023);
 minE = min([min(radiance_mapR(:)),min(radiance_mapG(:)),min(radiance_mapB(:))]);
 maxE = max([max(radiance_mapR(:)),max(radiance_mapG(:)),max(radiance_mapB(:))]);
 
%normalization
for k = 1:682 
   for l = 1:1023
         EnormR(k,l) = (radiance_mapR(k,l)-minE)/(maxE-minE);
         EnormG(k,l) = (radiance_mapG(k,l)-minE)/(maxE-minE);
         EnormB(k,l) = (radiance_mapB(k,l)-minE)/(maxE-minE);
   end
end

%%gamma correction
gamma = 0.5;
A = 0.75;
EgammaR = A*EnormR.^gamma;
EgammaG = A*EnormG.^gamma;
EgammaB = A*EnormB.^gamma;

imageGamma = cat(3,EgammaR,EgammaG,EgammaB);
figure
imshow(imageGamma)
title('Gamma')

Enorm3 = cat(3,EnormR,EnormG,EnormB);
 M = tonemap(Enorm3);


Rnew = zeros(682,1023);
Gnew = zeros(682,1023);
Bnew = zeros(682,1023);

for k3 = 1:682
    for k4 = 1:1023
        Rnew(k3,k4) = M(k3,k4)*EnormR(k3,k4);
        Gnew(k3,k4) = M(k3,k4)*EnormG(k3,k4);
        Bnew(k3,k4) = M(k3,k4)*EnormB(k3,k4);
    end
end

newI = cat(3,Rnew,Gnew,Bnew);
figure
imshow (newI)% M = Ltone./L1;
title ('Reinhard')
