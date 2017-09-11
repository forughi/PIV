%!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!%
%       Particle Image Velocimetry (PIV) code!           %
%      Developed by A. F. Forughi (March, 2014)          %
%   Mech. Eng Dept., The University of British Columbia  %
%!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!%

clc
clear all;
tic
% cd('C:\pictures') %Location

img1 = imread('a1.jpg'); %Input 1st frame
img2 = imread('a2.jpg'); %Input 2nd frame

icorr=0;            %Number of Correction Cycle
Rlimit=0.4;         %Minimum valid R:

iw=31; %Interrodation Windows Sizes (pixel)
sw=70; %Search Windows Sizes (sw > iw) (pixel)

iw=2*round(iw/2)-1; %Even->Odd
sw=2*round(sw/2)-1;

scale=1.0;  %spatial scale (unit = m/pixel)
tscale=1.0; %time step = 1/frame_rate (unit= s/frame)


%Converting RGB to Grayscale if need be:
sz=size(size(img1));
if sz(2)>=3 %RGB
    a1 = double(img1(:,:,1)+img1(:,:,2)+img1(:,:,3))/3.0;
    a2 = double(img2(:,:,1)+img2(:,:,2)+img2(:,:,3))/3.0;
    disp('RGB->GREYSCALE....Done!')
else   %GREYSCALE
    a1 = double(img1(:,:));
    a2 = double(img2(:,:));
end
a=flip(a1',2);b=flip(a2',2);[ia,ja] = size(a);


%Calculating Size of Matrices:
im=2*floor((ia-iw)/(iw-1)); %Number of I.W.s in x direction
jm=2*floor((ja-iw)/(iw-1)); %Number of I.W.s in y direction

vecx(1:im,1:jm)=0.;%x-Displacement
vecy(1:im,1:jm)=0.;%y-Displacement
vec(1:im,1:jm)=0.; %Magnification
rij(1:im,1:jm)=0.; %Correlation coeff.
R(1:sw-iw+1,1:sw-iw+1)=0.; %Correlation Matrix
margin=1;%floor((sw-iw)/floor(iw-1)+2);

for j=1:jm % ID of I.W. in j
    for i=1:im % ID of I.W. in i
        i1=(i-1)*(iw-1)/2+1;  %left bound
        ii1=i1+iw;  %right bound
        j1=(j-1)*(iw-1)/2+1;  %bottom bound
        jj1=j1+iw;  %top bound
        
        %Searching:
        R(1:sw-iw+1,1:sw-iw+1)=-1.0;
        for jj=1:sw-iw+1  % j-search
            for ii=1:sw-iw+1  % i-search
                if ((i1+ii-(sw-iw)/2-1>=1)&&(ii1+ii-(sw-iw)/2-1<=ia)&&(j1+jj-(sw-iw)/2-1>=1)&&(jj1+jj-(sw-iw)/2-1<=ja))
                    R(ii,jj)=corr2(b(i1+ii-(sw-iw)/2-1:ii1+ii-(sw-iw)/2-1,j1+jj-(sw-iw)/2-1:jj1+jj-(sw-iw)/2-1),a(i1:ii1,j1:jj1));
                end
            end
        end
        
        R(isnan(R)==1)=-1; %Removing NaN(s)
        %Finding Location of maximum R
        [maxR,Y]=max(max(R));   %Y:location of maximum R
        [maxR,X]=max(max(R'));  %X:location of maximum R
        
        if (maxR>=Rlimit)   %condition for removing unacceptable R coefficients
            if ((X<=1)||(Y<=1)||(X>=(sw-iw+1))||(Y>=(sw-iw+1)))
                vecx(i,j)=X-(sw-iw)/2-1;
                vecy(i,j)=Y-(sw-iw)/2-1;
            else
                c=(R(X+1,Y)+R(X-1,Y)-2*R(X,Y))*(R(X,Y+1)+R(X,Y-1)-2*R(X,Y)); %(zero check!)
                if (c==0.)
                    vecx(i,j)=X-(sw-iw)/2-1;
                    vecy(i,j)=Y-(sw-iw)/2-1;
                else
                    vecx(i,j)=X-(sw-iw)/2-1-(R(X+1,Y)-R(X-1,Y))/(2*(R(X+1,Y)+R(X-1,Y)-2*R(X,Y)));
                    vecy(i,j)=Y-(sw-iw)/2-1-(R(X,Y+1)-R(X,Y-1))/(2*(R(X,Y+1)+R(X,Y-1)-2*R(X,Y)));
                end
            end
        else    %Low R:
            vecx(i,j)=0.;
            vecy(i,j)=0.;
        end
        vec(i,j)=sqrt(vecx(i,j)^2+vecy(i,j)^2);
        rij(i,j)=maxR;
    end
    disp([num2str(floor(100*j/jm)),'% passed']); %Progress (%)
end
disp('Done!');

%% Bad Vectors:
disorder=0;
for ii=1:icorr %Correction Cycle
    for j=2:jm-1
        for i=2:im-1
            if ((((vecx(i+1,j)*vecx(i-1,j))>0.)&&((vecx(i+1,j)*vecx(i,j))<0.))||(rij(i,j)<Rlimit))
                vecx(i,j)=0.25*(vecx(i+1,j)+vecx(i-1,j)+vecx(i,j+1)+vecx(i,j-1));
                vecy(i,j)=0.25*(vecy(i+1,j)+vecy(i-1,j)+vecy(i,j+1)+vecy(i,j-1));
                disorder=disorder+1;
            end
            if (((vecy(i,j+1)*vecy(i,j-1))>0.)&&((vecy(i,j+1)*vecy(i,j))<0.)||(rij(i,j)<Rlimit))
                vecx(i,j)=0.25*(vecx(i+1,j)+vecx(i-1,j)+vecx(i,j+1)+vecx(i,j-1));
                vecy(i,j)=0.25*(vecy(i+1,j)+vecy(i-1,j)+vecy(i,j+1)+vecy(i,j-1));
                disorder=disorder+1;
            end
            vec(i,j)=sqrt(vecx(i,j)^2+vecy(i,j)^2);
        end
    end
end

%% Exporting:
% Applying spatiotemporal scales:
x(1:im,1:jm)=0.;y(1:im,1:jm)=0.;u(1:im,1:jm)=0.;v(1:im,1:jm)=0.;vel(1:im,1:jm)=0.;
for j=1:jm
    for i=1:im
        x(i,j)=(i*(iw-1)/2)*scale;
        y(i,j)=(j*(iw-1)/2)*scale;
        u(i,j)=vecx(i,j)*scale/tscale;
        v(i,j)=vecy(i,j)*scale/tscale;
        vel(i,j)=vec(i,j)*scale/tscale;
    end
end
figure
contourf(x,y,rij(:,:));
colormap(jet);
colorbar;
% hold on
figure
quiver (x, y, u, v,'black');

%Statistics:
disp(['IW= ', num2str(iw)]);
disp(['SW= ', num2str(sw)]);
disp('******************************************')
disp(['Rlimit= ', num2str(Rlimit)])
disp(['iCorr= ', num2str(icorr)]);
disp(['Total Correction number= ', num2str(disorder)]);
disp('******************************************')
disp(['R_mean= ', num2str(sum(sum(rij))/(im*jm))]);
disp(['R_max= ', num2str(max(max(rij)))]);
disp(['R_min= ', num2str(min(min(rij)))]);
disp('******************************************')
disp(['|v|_mean= ', num2str(sum(sum(vec))/(im*jm))]);
disp(['|v|_max= ', num2str(max(max(vec)))]);
disp(['|v|_min= ', num2str(min(min(vec)))]);
disp('******************************************')
disp(['vx_mean= ', num2str(sum(sum(vecx))/(im*jm))]);
disp(['vx_max= ', num2str(max(max(vecx)))]);
disp(['vx_min= ', num2str(min(min(vecx)))]);
disp('******************************************')
disp(['vy_mean= ', num2str(sum(sum(vecy))/(im*jm))]);
disp(['vy_max= ', num2str(max(max(vecy)))]);
disp(['vy_min= ', num2str(min(min(vecy)))]);
disp('******************************************')

% Exporting to tecplot (results.plt):
fid=fopen('result.plt','w');
fprintf(fid,'VARIABLES= X,Y,U,V,Vel,R\n');
fprintf(fid,'ZONE I=');
fprintf(fid,'%d',im);
fprintf(fid,',J=');
fprintf(fid,'%d',jm);
fprintf(fid,' F=POINT\n');
for j=1:jm
    for i=1:im
        fprintf(fid,'%f %f %f %f %f %f\n',x(i,j),y(i,j),u(i,j),v(i,j),vel(i,j),rij(i,j));
    end
end
fclose(fid);
toc
