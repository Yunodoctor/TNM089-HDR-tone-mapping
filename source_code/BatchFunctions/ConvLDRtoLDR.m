function ConvLDRtoLDR(fmtIn, fmtOut)
%
%        ConvLDRtoLDR(fmtIn, fmtOut)
%
%        This batch function converts LDR images in the current directory
%        from a format, fmtIn, to another LDR format, fmtOut.
%        
%        For example:
%           ConvLDRtoLDR('jpg','png');
%
%        This lines converts .jpg files in the folder into .png files
%
%        Input:
%           -fmtIn: an input string represeting the LDR format of the images
%           to be converted. This can be: 'jpeg', 'jpg', 'png', etc.
%           -fmtOut: an input string represeting the LDR format of
%           converted images. This can be: 'jpeg', 'jpg', 'png', etc.
%
%     Copyright (C) 2012-15  Francesco Banterle
%
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.
%

lst = dir(['*.', fmtIn]);

for i=1:length(lst)
    tmp_name = lst(i).name;
    disp(tmp_name);
    
    img = imread(tmp_name);
    tmp_name_we = RemoveExt(tmp_name);
    imwrite(img,[tmp_name_we, '.', fmtOut]);
end

end