function imgBlur = filterBox(img, radius)
%
%
%       imgBlur = filterBox(img, radius)
%
%
%       Input:
%           -img: the input image
%           -radius: the radius of the box filter
%
%       Output:
%           -imgBlur: a filtered image
%
%
%     Copyright (C) 2011-15 Francesco Banterle
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

if(~exist('radius', 'var'))
    radius = 1;
end

if(radius < 1)
    radius = 1;
end

H = fspecial('average', round(radius * 2 + 1));
imgBlur = imfilter(img, H, 'replicate');

end

