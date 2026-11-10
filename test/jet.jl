@testitem "JET" begin
    using JET, NonconvexCore
    JET.test_package(NonconvexCore; target_modules=(NonconvexCore,))
end
