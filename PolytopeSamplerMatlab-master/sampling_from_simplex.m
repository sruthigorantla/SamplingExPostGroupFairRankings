
function s = sampling_from_simplex(d,k,lb,ub,num_samples)
	initSampler
	P = struct; 
	P.Aeq = ones(1, d);
	P.beq = 0;
	P.lb = lb;
	P.ub = ub;
	o = sample(P, num_samples); 
	s = o.samples;
	% scatter3(s(1,:),s(2,:),s(3,:));
	% scatter(s(1,:),s(2,:));
	% scatter(s(2,:),s(3,:));
	% scatter(s(1,:),s(3,:));
