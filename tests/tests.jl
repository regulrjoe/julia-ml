module Tests

include("../src/linear_regression.jl")
include("../src/logistic_regression.jl")
include("../src/gradient_descent.jl")
include("../src/feature_scaling.jl")

using Base.Test, MatrixDepot

export run

function run()
    test_linearreg()
    test_logisticreg()
    test_gradientdescent()
    test_featurescaling()
end

# Test functions in linear_regression.jl
function test_linearreg()
    lr = LinearRegression
    @testset "Linear Regression Tests" begin
        @testset "Cost Function Unit Tests" begin
            @test lr.J(
                [1.0 2.0; 1.0 3.0; 1.0 4.0; 1.0 5.0],
                [7.0; 6.0; 5.0; 4.0],
                [0.1; 0.2]
            ) ≈ 11.945 atol = 0.0001
            @test lr.J(
                [1.0 2.0 3.0; 1.0 3.0 4.0; 1.0 4.0 5.0; 1.0 5.0 6.0],
                [7.0; 6.0; 5.0; 4.0],
                [0.1; 0.2; 0.3]
            ) ≈ 7.0175 atol = 0.0001
            @test lr.J(
                [2.0 1.0 3.0; 7.0 1.0 9.0; 1.0 8.0 1.0; 3.0 7.0 4.0],
                [2.0; 5.0; 5.0; 6.0],
                [0.4; 0.6; 0.8]
            ) ≈ 5.2950 atol = 0.0001
        end
        @testset "Normal Equation Unit Tests" begin
            @test lr.normal_equation(
                [2.0 1.0 3.0; 7.0 1.0 9.0; 1.0 8.0 1.0; 3.0 7.0 4.0],
                [2.0; 5.0; 5.0; 6.0]
            ) ≈ [0.0083857; 0.5681342; 0.4863732] atol = 0.0001
        end
    end
    return nothing
end

# Test functions in logistic_regression.jl
function test_logisticreg()
    lr = LogisticRegression
    @testset "Logistic Regression Tests" begin
        @testset "Cost Function Unit Tests" begin
            G = [0.0; 0.0; 0.0; 0.0]
            output = lr.J!(
                [ones(3,1) matrixdepot("magic", Float64, 3)],
                [1.0; 0.0; 1.0],
                [-2.0; -1.0; 1.0; 2.0],
                G = G
            )
            @test output ≈ 4.6832 atol = 0.0001
            @test G ≈ [0.31722; 0.87232; 1.64812; 2.23787] atol = 0.0001

            output = lr.J!(
                [ones(3,1) matrixdepot("magic", Float64, 3)],
                [1.0; 0.0; 1.0],
                [-2.0; -1.0; 1.0; 2.0],
                G = G,
                reg = 4
            )
            @test output ≈ 8.6832 atol = 0.0001
            @test G ≈ [0.31722; -0.46102; 2.98146; 4.90454] atol = 0.0001
        end
        @testset "Sigmoid Function Unit Tests" begin
            @test lr.g(0.0) == 0.5
            @test lr.g([0.0, 0.0]) == [0.5, 0.5]
            @test lr.g([0.0 0.0; 0.0 0.0]) == [0.5 0.5; 0.5 0.5]
            @test lr.g(-5) ≈ 0.0066929 atol = 0.0001
            @test lr.g(5) ≈ 0.99331 atol = 0.0001
            @test lr.g([4.0; 5.0; 6.0]) ≈ [0.98201; 0.99331; 0.99753] atol = 0.0001
            @test lr.g([-1.0; 0.0 ;1.0]) ≈  [0.26894; 0.50000; 0.73106] atol = 0.0001
            @test sum(lr.g(reshape(-1:0.1:0.9, 4, 5))) ≈ 9.76894 atol = 0.0001
        end
        @testset "Prediction Function Unit Tests" begin
            @test round.(lr.predict(
                [1.0 1.0; 1.0 2.5; 1.0 3.0; 1.0 4.0],
                [-3.5 ; 1.3]
            )) ≈ [0.0; 0.0; 1.0; 1.0] atol = 0.0001
        end
    end
    return nothing
end

# Test functions in gradient_descent.jl
function test_gradientdescent()
    gd = GradientDescent
    @testset "Gradient Descent Tests" begin
        @testset "Linear Regression Unit Tests" begin
            @test gd.gradient_descent(
                [1.0 5.0; 1.0 2.0; 1.0 4.0; 1.0 5.0],
                [1.0; 6.0; 4.0; 2.0],
                LinearRegression.h,
                LinearRegression.J,
                config = gd.GDConfig(0.01, 0.0001, 0.0, 1000)
            ) ≈ [5.2147; -0.573346] atol = 0.0001
            @test gd.gradient_descent(
                [2.0 1.0 3.0; 7.0 1.0 9.0; 1.0 8.0 1.0; 3.0 7.0 4.0],
                [2.0; 5.0; 5.0; 6.0],
                LinearRegression.h,
                LinearRegression.J,
                config = gd.GDConfig(0.01, 0.0001, 0.0, 10)
            ) ≈ [0.25175; 0.53779; 0.32282] atol = 0.0001
        end
        @testset "Logistic Regression Unit Tests" begin

        end
    end
    return nothing
end

# Test functions in feature_scaling.jl
function test_featurescaling()
    fs = FeatureScaling
    @testset "Feature Scaling Unit Tests" begin
        X = [1.0; 2.0; 3.0]
        params = fs.fnormalize!(X)
        @test X ≈ [-1.0; 0.0; 1.0] atol = 0.0001
        @test params.mu ≈ [2.0] atol = 0.0001
        @test params.sd ≈ [1.0] atol = 0.0001

        X = matrixdepot("magic", Float64, 3)
        params = fs.fnormalize!(X)
        @test X ≈ [1.13389 -1.00000 0.37796;
            -0.75593 0.00000 0.75593;
            -0.37796 1.00000 -1.13389] atol = 0.0001
        @test params.mu ≈ [5.0; 5.0; 5.0] atol = 0.0001
        @test params.sd ≈ [2.6458; 4.0000; 2.6458] atol = 0.0001

        X = [-ones(1,3); matrixdepot("magic", Float64, 3)]
        params = fs.fnormalize!(X)
        @test X ≈ [-1.21725  -1.01472  -1.21725;
            1.21725 -0.56373 0.67625;
            -0.13525 0.33824 0.94675;
            0.13525 1.24022 -0.40575] atol = 0.0001
        @test params.mu ≈ [3.5; 3.5; 3.5] atol = 0.0001
        @test params.sd ≈ [3.6968; 4.4347; 3.6968] atol = 0.0001
    end
    return nothing
end

end
