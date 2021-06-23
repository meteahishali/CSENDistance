function [aa] = solve_dalm(y,D)
    aa=SolveDALM(D, y,'tolerance',0.01);
end