function a_pinv = t_pinv(a)
    % tubal transpose  of 3d-tensor
    
    if ndims(a) ~= 3
       error("dimension of tensor a should be equal 3")
    end
    
    [n1, n2, n3] = size(a);
    a_pinv = zeros([n2, n1, n3]);
    
    a = fft(a, [], 3);
    for i=1:n3
        a_pinv(:,:,i) = pinv(a(:,:,i));
    end   
    a_pinv = ifft(a_pinv, [], 3);
end