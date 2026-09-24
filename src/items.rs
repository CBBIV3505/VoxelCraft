//! Items: everything that can sit in a hotbar slot. Blocks are the existing
//! [`crate::world::BlockType`]s; tools (wooden pickaxe / axe) and sticks are
//! new non-placeable item kinds. Also hosts break-time math (hold-to-mine),
//! crafting recipes, and the pixel-icon drawlists for the hotbar.

use crate::world::BlockType;

/// One hotbar slot's content: a stack of some item type.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ItemType {
    Block(BlockType),
    Stick,
    WoodPickaxe,
    WoodAxe,
    StonePickaxe,
}

impl ItemType {
    pub fn name(self) -> &'static str {
        match self {
            ItemType::Block(t) => t.name(),
            ItemType::Stick => "stick",
            ItemType::WoodPickaxe => "wood pick",
            ItemType::WoodAxe => "wood axe",
            ItemType::StonePickaxe => "stone pick",
        }
    }

    /// Hotbar icon color(s): [fill, accent] in linear RGB. Tools use the
    /// accent for their head/blade pixels; sticks are all wood.
    #[allow(dead_code)] // superseded by overlay.rs's local bitmaps; kept for tests
    pub fn icon_colors(self) -> ([f32; 3], [f32; 3]) {
        fn lin(c: u8) -> f32 {
            let c = c as f32 / 255.0;
            if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        }
        match self {
            ItemType::Block(t) => (fill_for_block(t), fill_for_block(t)),
            ItemType::Stick => ([lin(150), lin(116), lin(70)], [lin(150), lin(116), lin(70)]),
            ItemType::WoodPickaxe | ItemType::WoodAxe => (
                [lin(146), lin(112), lin(66)], // wooden handle
                [lin(178), lin(140), lin(88)], // slightly lighter head
            ),
            ItemType::StonePickaxe => (
                [lin(146), lin(112), lin(66)],
                [lin(128), lin(128), lin(132)], // gray stone head
            ),
        }
    }

    /// What this item looks like as a world drop (floating mini-cube color).
    #[allow(dead_code)] // superseded by overlay.rs's local bitmaps; kept for tests
    pub fn drop_color(self) -> [f32; 3] {
        match self {
            ItemType::Block(t) => fill_for_block(t),
            ItemType::Stick => [0.28, 0.21, 0.12],
            ItemType::WoodPickaxe | ItemType::WoodAxe => [0.34, 0.26, 0.15],
            ItemType::StonePickaxe => [0.35, 0.35, 0.38],
        }
    }
}

/// Block fill colors shared with overlay.rs (linear RGB of the sRGB palette).
#[allow(dead_code)] // superseded by overlay.rs's local bitmaps
pub fn fill_for_block(t: BlockType) -> [f32; 3] {
    fn chan(c: u8) -> f32 {
        let c = c as f32 / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }
    match t {
        BlockType::Grass => [chan(110), chan(168), chan(68)],
        BlockType::Dirt => [chan(134), chan(96), chan(67)],
        BlockType::Stone => [chan(125), chan(125), chan(125)],
        BlockType::Log => [chan(109), chan(84), chan(50)],
        BlockType::Leaves => [chan(60), chan(124), chan(45)],
        BlockType::Planks => [chan(168), chan(133), chan(84)],
        BlockType::CraftingTable => [chan(150), chan(106), chan(62)],
        BlockType::CoalOre => [chan(58), chan(58), chan(62)],
        // Water is never an item; magenta sentinel like Air.
        BlockType::Water => [1.0, 0.0, 1.0],
        BlockType::Air => [1.0, 0.0, 1.0],
    }
}

// ---------------------------------------------------------------------------
// Mining speed
// ---------------------------------------------------------------------------

/// Base seconds to break a block bare-handed. Stone is the wall this whole
/// feature exists to break through (MC: 7.5 s hand vs 1.15 s wooden pick).
pub fn break_time(block: BlockType, held: Option<ItemType>) -> f32 {
    use BlockType as B;
    let hardness: f32 = match block {
        B::Grass => 0.45,
        B::Dirt => 0.4,
        B::Leaves => 0.25,
        B::Log => 1.6,
        B::Planks => 1.4,
        B::Stone => 5.0,
        B::CoalOre => 6.5,
        _ => 1.0,
    };
    let speed: f32 = match (block, held) {
        // Pickaxes speed up stone-family blocks; axes speed up wood.
        (B::Stone | B::CoalOre, Some(ItemType::StonePickaxe)) => 11.0,
        (B::Stone | B::CoalOre, Some(ItemType::WoodPickaxe)) => 7.0,
        (B::Log | B::Planks, Some(ItemType::WoodAxe)) => 5.0,
        _ => 1.0,
    };
    hardness / speed
}

/// Mining-tier gate: `None` = breakable by anything; `Some(msg)` = needs a
/// pickaxe, and bare hands/axes produce NOTHING (the block won't break).
/// Wood-tier picks break stone + coal; this is the Minecraft rule set.
pub fn tier_block_reason(block: BlockType, held: Option<ItemType>) -> Option<&'static str> {
    use BlockType as B;
    let needs_pick = matches!(block, B::Stone | B::CoalOre);
    if !needs_pick {
        return None;
    }
    match held {
        Some(ItemType::WoodPickaxe) | Some(ItemType::StonePickaxe) => None,
        _ => Some("needs a pickaxe"),
    }
}

// ---------------------------------------------------------------------------
// Crafting
// ---------------------------------------------------------------------------

/// A shaped recipe for the on-screen crafting grid. `pattern` is a compact
/// 3×3 row-major template; only the top-left `width × height` cells are used.
/// The matcher allows the pattern to be placed anywhere in the active 2×2 or
/// 3×3 grid, while every other cell must be empty.
#[derive(Clone, Copy, Debug)]
pub struct CraftRecipe {
    pub name: &'static str,
    pub width: usize,
    pub height: usize,
    pub pattern: [Option<ItemType>; 9],
    pub output: (ItemType, u32),
    pub key: char,
    pub requires_table: bool,
}

pub const CRAFT_RECIPES: [CraftRecipe; 6] = [
    CraftRecipe {
        name: "planks",
        width: 1,
        height: 1,
        pattern: [
            Some(ItemType::Block(BlockType::Log)),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ],
        output: (ItemType::Block(BlockType::Planks), 4),
        key: 'c',
        requires_table: false,
    },
    CraftRecipe {
        name: "sticks",
        width: 1,
        height: 2,
        pattern: [
            Some(ItemType::Block(BlockType::Planks)),
            None,
            None,
            Some(ItemType::Block(BlockType::Planks)),
            None,
            None,
            None,
            None,
            None,
        ],
        output: (ItemType::Stick, 4),
        key: 'r',
        requires_table: false,
    },
    CraftRecipe {
        name: "crafting table",
        width: 2,
        height: 2,
        pattern: [
            Some(ItemType::Block(BlockType::Planks)),
            Some(ItemType::Block(BlockType::Planks)),
            None,
            Some(ItemType::Block(BlockType::Planks)),
            Some(ItemType::Block(BlockType::Planks)),
            None,
            None,
            None,
            None,
        ],
        output: (ItemType::Block(BlockType::CraftingTable), 1),
        key: 'b',
        requires_table: false,
    },
    CraftRecipe {
        name: "wood pickaxe",
        width: 3,
        height: 3,
        pattern: [
            Some(ItemType::Block(BlockType::Planks)),
            Some(ItemType::Block(BlockType::Planks)),
            Some(ItemType::Block(BlockType::Planks)),
            None,
            Some(ItemType::Stick),
            None,
            None,
            Some(ItemType::Stick),
            None,
        ],
        output: (ItemType::WoodPickaxe, 1),
        key: 't',
        requires_table: true,
    },
    CraftRecipe {
        name: "wood axe",
        width: 3,
        height: 3,
        pattern: [
            Some(ItemType::Block(BlockType::Planks)),
            Some(ItemType::Block(BlockType::Planks)),
            None,
            Some(ItemType::Block(BlockType::Planks)),
            Some(ItemType::Stick),
            None,
            None,
            Some(ItemType::Stick),
            None,
        ],
        output: (ItemType::WoodAxe, 1),
        key: 'v',
        requires_table: true,
    },
    CraftRecipe {
        name: "stone pickaxe",
        width: 3,
        height: 3,
        pattern: [
            Some(ItemType::Block(BlockType::Stone)),
            Some(ItemType::Block(BlockType::Stone)),
            Some(ItemType::Block(BlockType::Stone)),
            None,
            Some(ItemType::Stick),
            None,
            None,
            Some(ItemType::Stick),
            None,
        ],
        output: (ItemType::StonePickaxe, 1),
        key: 's',
        requires_table: true,
    },
];

/// Return the ingredient stacks required by a recipe, with duplicate items
/// merged. This is the single source of truth used by the grid UI and the
/// keyboard quick-craft path; adding a shaped recipe cannot silently create a
/// different shortcut recipe.
pub fn recipe_ingredients(recipe: &CraftRecipe) -> Vec<(ItemType, u32)> {
    let mut ingredients = Vec::new();
    for y in 0..recipe.height {
        for x in 0..recipe.width {
            let Some(item) = recipe.pattern[y * 3 + x] else {
                continue;
            };
            if let Some((_, count)) = ingredients.iter_mut().find(|(ty, _)| *ty == item) {
                *count += 1;
            } else {
                ingredients.push((item, 1));
            }
        }
    }
    ingredients
}

/// Compact ingredient text for the recipe discovery panel. Singular item
/// labels keep the longest tool recipe inside the narrow panel.
#[allow(dead_code)] // panel list removed; kept for future recipe books
pub fn recipe_ingredients_text(recipe: &CraftRecipe) -> String {
    recipe_ingredients(recipe)
        .into_iter()
        .map(|(item, count)| {
            let mut name = item.name().to_uppercase();
            if count != 1 && !name.ends_with('S') {
                name.push('S');
            }
            format!("{count} {name}")
        })
        .collect::<Vec<_>>()
        .join(" + ")
}

/// Return the first shaped recipe matching the active grid, if any.
pub fn matching_recipe(
    grid: &[(Option<ItemType>, u32); 9],
    size: usize,
    table_available: bool,
) -> Option<&'static CraftRecipe> {
    let size = size.clamp(2, 3);
    CRAFT_RECIPES.iter().find(|recipe| {
        if recipe.requires_table && !table_available {
            return false;
        }
        if recipe.width > size || recipe.height > size {
            return false;
        }
        for oy in 0..=size - recipe.height {
            for ox in 0..=size - recipe.width {
                let mut fits = true;
                for y in 0..size {
                    for x in 0..size {
                        let expected = if x >= ox
                            && x < ox + recipe.width
                            && y >= oy
                            && y < oy + recipe.height
                        {
                            recipe.pattern[(y - oy) * 3 + (x - ox)]
                        } else {
                            None
                        };
                        let actual = grid[y * 3 + x].0;
                        if expected != actual {
                            fits = false;
                            break;
                        }
                    }
                    if !fits {
                        break;
                    }
                }
                if fits {
                    return true;
                }
            }
        }
        false
    })
}

// ---------------------------------------------------------------------------
// Hotbar pixel icons for non-block items (5×5 cell bitmaps, drawn filled)
// ---------------------------------------------------------------------------

/// 5×5 cell bitmap (rows top→bottom, 1 = filled cell) for non-block items.
#[allow(dead_code)] // superseded by overlay.rs's local bitmaps
pub fn item_bitmap(item: ItemType) -> [[u8; 5]; 5] {
    match item {
        // Diagonal stick, lower-left → upper-right.
        ItemType::Stick => [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
        ],
        // Pickaxe: curved head across the top, diagonal handle to bottom-left.
        ItemType::WoodPickaxe => [
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [1, 0, 0, 0, 0],
        ],
        // Axe: blade on the upper-left, diagonal handle to bottom-right.
        ItemType::WoodAxe => [
            [0, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        // Blocks keep their 3D cube icon — this fn is only called for tools.
        _ => [[0; 5]; 5],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use BlockType as B;

    #[test]
    fn tools_speed_up_their_blocks() {
        // Bare hands: stone is slow, wood is medium, dirt fast.
        let (stone, wood, dirt) = (
            break_time(B::Stone, None),
            break_time(B::Log, None),
            break_time(B::Dirt, None),
        );
        assert_eq!(stone, 5.0);
        assert_eq!(wood, 1.6);
        assert_eq!(dirt, 0.4);

        // Wooden pickaxe: 7× on stone, no effect on wood.
        assert!((break_time(B::Stone, Some(ItemType::WoodPickaxe)) - 5.0 / 7.0).abs() < 1e-5);
        assert_eq!(break_time(B::Log, Some(ItemType::WoodPickaxe)), wood);

        // Wooden axe: 5× on wood products, no effect on stone.
        assert!((break_time(B::Log, Some(ItemType::WoodAxe)) - 1.6 / 5.0).abs() < 1e-5);
        assert_eq!(break_time(B::Stone, Some(ItemType::WoodAxe)), stone);
    }

    #[test]
    fn stone_tier_gates_and_speeds() {
        // Stone/coal are impossible bare-handed and with an axe, possible
        // with any pickaxe.
        assert_eq!(tier_block_reason(B::Stone, None).is_some(), true);
        assert_eq!(
            tier_block_reason(B::CoalOre, Some(ItemType::WoodAxe)).is_some(),
            true
        );
        assert_eq!(
            tier_block_reason(B::Stone, Some(ItemType::WoodPickaxe)),
            None
        );
        assert_eq!(
            tier_block_reason(B::CoalOre, Some(ItemType::StonePickaxe)),
            None
        );
        // Dirt/wood remain ungated.
        assert_eq!(tier_block_reason(B::Dirt, None), None);
        // Stone pick mines stone faster than wood pick.
        assert!(
            break_time(B::Stone, Some(ItemType::StonePickaxe))
                < break_time(B::Stone, Some(ItemType::WoodPickaxe))
        );
    }

    #[test]
    fn stone_pickaxe_recipe_matches_in_the_table_grid() {
        let mut grid = [(None, 0); 9];
        let stone = ItemType::Block(B::Stone);
        grid[0] = (Some(stone), 1);
        grid[1] = (Some(stone), 1);
        grid[2] = (Some(stone), 1);
        grid[4] = (Some(ItemType::Stick), 1);
        grid[7] = (Some(ItemType::Stick), 1);
        let r = matching_recipe(&grid, 3, true).unwrap();
        assert_eq!(r.name, "stone pickaxe");
        assert!(r.requires_table);
    }

    #[test]
    fn canonical_recipes_have_expected_ingredients_and_keys() {
        let by_key = |key: char| {
            CRAFT_RECIPES
                .iter()
                .find(|recipe| recipe.key == key)
                .unwrap()
        };
        let ingredients = |key: char| recipe_ingredients(by_key(key));
        assert_eq!(ingredients('c'), vec![(ItemType::Block(B::Log), 1)]);
        assert_eq!(by_key('c').output, (ItemType::Block(B::Planks), 4));
        assert_eq!(ingredients('r'), vec![(ItemType::Block(B::Planks), 2)]);
        assert_eq!(by_key('r').output, (ItemType::Stick, 4));
        assert_eq!(ingredients('b'), vec![(ItemType::Block(B::Planks), 4)]);
        assert_eq!(by_key('b').output, (ItemType::Block(B::CraftingTable), 1));
        assert_eq!(
            ingredients('t'),
            vec![(ItemType::Block(B::Planks), 3), (ItemType::Stick, 2),]
        );
        assert_eq!(ingredients('v'), ingredients('t'));
        assert!(by_key('t').requires_table);
        assert!(by_key('v').requires_table);
        assert_eq!(recipe_ingredients_text(by_key('t')), "3 PLANKS + 2 STICKS");
    }

    #[test]
    fn shaped_grid_uses_two_by_two_for_the_table_and_three_by_three_for_tools() {
        let mut grid = [(None, 0); 9];
        let planks = ItemType::Block(B::Planks);
        grid[0] = (Some(planks), 1);
        grid[1] = (Some(planks), 1);
        grid[3] = (Some(planks), 1);
        grid[4] = (Some(planks), 1);
        assert_eq!(
            matching_recipe(&grid, 2, false).unwrap().name,
            "crafting table"
        );
        grid = [(None, 0); 9];
        grid[0] = (Some(planks), 1);
        grid[1] = (Some(planks), 1);
        grid[2] = (Some(planks), 1);
        grid[4] = (Some(ItemType::Stick), 1);
        grid[7] = (Some(ItemType::Stick), 1);
        assert_eq!(
            matching_recipe(&grid, 3, true).unwrap().name,
            "wood pickaxe"
        );
        assert!(matching_recipe(&grid, 2, true).is_none());
    }
}
