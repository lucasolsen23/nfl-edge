"""Player-props projection engine (Phase 4).

Projects a player's stat (passing/rushing/receiving yards, receptions) for an
upcoming game from walk-forward recent form adjusted for the opponent defense,
producing a mean AND a spread so any prop line can be priced as P(over).

Utah context: player props on PrizePicks / Underdog are legal and softer than
game markets -- this is the most promising edge. The projection is the model; a
line source (manual entry, book props, or PrizePicks) is plugged in to score it.
"""
