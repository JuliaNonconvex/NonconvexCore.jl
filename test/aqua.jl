@testitem "Aqua" begin
    using Aqua, NonconvexCore, ChainRulesCore
    Aqua.test_all(
        NonconvexCore;
        # The TransformVariables transforms get ChainRules rrules for
        # use with Zygote — deliberate extension of `rrule`.
        piracies=(treat_as_own=[rrule],),
    )
end
