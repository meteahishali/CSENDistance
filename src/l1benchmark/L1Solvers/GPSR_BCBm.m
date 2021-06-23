function [x,totIter]= GPSR_BCB(y,A,tau,blocksize,verbose)
	% Set the defaults for the optional parameters
	tolA = 0.01;
	maxiter = 10000;
	%maxiter = 1000;
	miniter = 5;
	alphamin = 1e-30;
	alphamax = 1e30;
	
	if ~exist('blocksize', 'var')
		blocksize = 32; %32*3 seems good
	end
	if ~exist('verbose', 'var')
		verbose = 1;
	end

   	dims = min(blocksize,size(y,2));
    len = size(y,2);
    Y = y;
    y = Y(:,1:dims);
    todoIdx = 1:len;
    doneIdx = zeros(len,1);
    inQ = zeros(dims,1);
    X = sparse(size(A,2), len);
	totIter = zeros(size(A,2), 1);
    
	% Precompute A'*y since it'll be used a lot
	%Aty = A'*y;
	
	% Initialization
	xB = zeros(size(A,2),dims);%AT(zeros(size(y)));

	% initialize u and v
	uB = zeros(size(xB));%x.*(x >= 0);
	vB = zeros(size(xB));%-x.*(x <  0);

	% store given stopping criterion and threshold, because we're going 
	% to change them in the continuation procedure
	final_tolA = tolA;
	iter = ones(dims,1);

	% Compute and store initial value of the objective function
	%resid =  y - A*xB;

	tolA = final_tolA;
	%alphaB = ones(1, dims);

	% Compute the initial gradient and the useful 
	% quantity resid_base
	%resid_base = y - resid;

	% control variable for the outer loop and iteration counter
	keep_going = ones(dims, 1);
    if verbose
        fprintf(' %0.6d/%0.6d', sum(doneIdx), len);
    end
	while sum(doneIdx == 0) %sum(keep_going)

        % See if a slot opened up and we have more to process
        if sum(inQ == 0)% && ~isempty(todoIdx)
            i = 1;
			while i <= size(inQ,1)
                if ~inQ(i)
					if isempty(todoIdx)
						% Remove from queue
						y(:,i) = [];
						xB(:,i) = [];
						uB(:,i) = [];
						vB(:,i) = [];
						iter(i) = [];
						alphaB(i) = [];
						keep_going(i) = [];
						inQ(i) = [];
					else
						idx = todoIdx(end);
						todoIdx(end) = [];
						inQ(i) = idx;

						y(:,i) = Y(:,idx);
						xB(:,i) = 0;
						uB(:,i) = 0;
						vB(:,i) = 0;
						iter(i) = 1;
						alphaB(i) = 1;
						keep_going(i) = 1;
						i = i + 1;
					end
				else
					i = i + 1;
                end
				
            end
            
            Aty = A'*y;
            %resid =  y - A*xB;
            %resid_base = y - resid;
            resid_base = A*xB;
            %resid = y - resid_base;
        end
        
		% compute gradient
		tempB = A'*resid_base;

		termB  =  tempB - Aty;
		graduB =  termB + tau;
		gradvB =  tau - termB ;

		% projection and computation of search direction vector
%         for k = 1:dims
%             duB(:,k) = max(uB(:,k) - alphaB(k).*graduB(:,k), 0.0) - uB(:,k);
%             dvB(:,k) = max(vB(:,k) - alphaB(k).*gradvB(:,k), 0.0) - vB(:,k);
%         end
		ALPHA = repmat(alphaB, size(graduB,1), 1);
		duB = max(uB - ALPHA.*graduB, 0.0) - uB;
		dvB = max(vB - ALPHA.*gradvB, 0.0) - vB;
		%assert(min(min(max(uB - ALPHA.*graduB, 0.0))) >= 0);
		%assert(min(min(max(vB - ALPHA.*gradvB, 0.0))) >= 0);
        
		dxB = duB-dvB;
        
		% calculate useful matrix-vector product involving dx
		auvB = A*dxB;
        
		old_uB = uB; 
		old_vB = vB;

		% Everything after this can be individualized
		for k = 1:length(keep_going)
			
			if ~keep_going(k)
				continue
			end
			
% 			temp = tempB(:,k);
% 			term = termB(:,k);
% 			gradu = graduB(:,k);
% 			gradv = gradvB(:,k);
% 			du = duB(:,k);
% 			dv = dvB(:,k);
% 			old_u = old_uB(:,k);
% 			old_v = old_vB(:,k);
%			auv = auvB(:,k);
% 			u = uB(:,k);
% 			v = vB(:,k);
% 			x = xB(:,k);
			alpha = alphaB(:,k);

			dGd = auvB(:,k)'*auvB(:,k);

			% monotone variant: calculate minimizer along the direction (du,dv)
			lambda0 = - (graduB(:,k)'*duB(:,k) + gradvB(:,k)'*dvB(:,k))/(realmin+dGd);
			if lambda0 < 0
				fprintf(' ERROR: lambda0 = %10.3e negative. Quit\n', lambda0);
				return;
			end
			lambda = min(lambda0,1);

			uB(:,k) = old_uB(:,k) + lambda * duB(:,k);
			vB(:,k) = old_vB(:,k) + lambda * dvB(:,k);
% 			uvmin = min(u,v);
%             if min(uvmin(:)) ~= max(uvmin(:))
%                x = 1;
%             end
% 			uB(:,k) = uB(:,k) - uvmin; 
% 			vB(:,k) = vB(:,k) - uvmin; 
			xB(:,k) = uB(:,k) - vB(:,k);

			% compute new alpha
			dd  = duB(:,k)'*duB(:,k) + dvB(:,k)'*dvB(:,k);  
			if dGd <= 0
				% something wrong if we get to here
				%fprintf(1,' dGd=%12.4e, nonpositive curvature detected\n', dGd);
				alphaB(:,k) = alphamax;
			else
				alphaB(:,k) = min(alphamax,max(alphamin,dd/dGd));
			end

			% ----- Core update ------
			resid_base(:,k) = resid_base(:,k) + lambda*auvB(:,k); 
			%uB(:,k) = u;
			%vB(:,k) = v;
			%xB(:,k) = x;
			%alphaB(:,k) = alpha;
			
			%history = [history x];

			% update iteration counts, store results and times
			iter(k) = iter(k) + 1;

			% compute the "LCP" stopping criterion - again based on the previous
			% iterate. Make it "relative" to the norm of x.
			w = [ min(graduB(:,k), old_uB(:,k)); min(gradvB(:,k), old_vB(:,k)) ];
			criterionLCP = norm(w(:), inf);
			criterionLCP = criterionLCP / max([1.0e-6, norm(old_uB(:,k),inf), norm(old_vB(:,k),inf)]);
			keep_going(k) = (criterionLCP > tolA);

			% take no less than miniter... 
			if iter(k)<=miniter
				keep_going(k) = 1;
			elseif iter(k) > maxiter %and no more than maxiter iterations  
				keep_going(k) = 0;
            end
            
            % Check if we are done with this one
            if ~keep_going(k)
                idx = inQ(k);
                inQ(k) = 0; % Flag empty spot
                X(:,idx) = xB(:,k);
                doneIdx(idx) = 1;
				totIter(idx) = iter(k);
				
                if verbose
                    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b%0.6d/%0.6d', sum(doneIdx), len);
                end
            end
		end
	end % end of the main loop of the BB-QP algorithm
	
	x = X;
end