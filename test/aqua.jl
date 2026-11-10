@testitem "Aqua" begin
    using Aqua, NonconvexCore
    Aqua.test_all(NonconvexCore)
end
