# UI/UX Design Guidelines

Distilled from video transcripts on UI/UX fundamentals, common beginner mistakes, and design principles for top-tier websites.

---

## 1. Signifiers: Make the UI Explain Itself

The UI should communicate how it works without instructions. Users should just know.

- Containers group related items and imply relationship
- Active/selected states (background highlight, border) tell users where they are
- Grayed-out text signals an item is inactive or disabled
- Hover states, button press states, tooltips, and active nav highlights are all signifiers
- If you have to write instructions on how something works, the design failed first

---

## 2. Visual Hierarchy

Use size, position, and color to rank importance. Flat, uniform content reads like a spreadsheet.

- Most important content: top, large, bold, or colored
- Secondary content: below, smaller, lower contrast
- Images at the top of a card add instant visual weight and make scanning easy
- Contrast between elements (big vs small, colorful vs muted) IS the hierarchy
- Use icons with alignment instead of text labels when the meaning is obvious (e.g. an arrow between two locations)
- To emphasize something, de-emphasize competing elements. Reduce the contrast or weight of secondary information -- that creates emphasis without making the primary element louder.
- After completing a design, zoom out and scan it in 2-3 seconds. If the primary element does not stand out immediately, the hierarchy is not working.
- Not all H1 tags look the same. HTML tags are semantic. Visual size and weight are context-dependent. A H3 can be larger than a H2 if that is what the layout calls for.

---

## 3. Spacing and White Space

White space is more important than grids. Cramped layouts feel amateur.

- Use 32px between major sections as a starting point
- Group closely related elements (heading + subtext) with tighter spacing, separate unrelated sections with more
- Use the 4-point grid system (multiples of 4) for spacing values -- not because it looks better but because values can always be halved, which keeps things consistent. A practical scale to use: 4 8 12 16 20 28 40 60 100 160 240
- On mobile, you always need more space than you think
- 12-column grids are guidelines, not rules. Custom landing pages often break the grid and that is fine. Galleries and repeating content benefit more from grid discipline.
- Start with too much spacing, then reduce. When you are focused on one element the space looks excessive, but users scan the whole layout first. Give it room, then tighten until it feels right.
- Use REM units for spacing and font sizes so the layout respects the user's system font size preference. To convert: divide the pixel value by 16 (e.g. 32px = 2rem).
- Define spacing as CSS variables or design tokens. Pick from a fixed scale rather than choosing arbitrary values on the fly.

---

## 4. Typography

Design is mostly text. Treat typography deliberately.

- One font is almost always enough. Pick a good sans-serif and stick to it.
- For large display/hero headings: tighten letter spacing to around -2% to -3%, and drop line height to 110-120%. This alone makes headers look polished.
- Cap font sizes: no more than 6 distinct sizes on a landing page. On dashboards, rarely go above 24px because information density is higher.
- Hierarchy comes from contrast between sizes, not from using multiple fonts.
- Line height is inversely proportional to font size. Smaller text needs a larger line height (more space between lines) for legibility. Large headings can have tighter line height.
- Line height on text elements acts as built-in top margin. You often do not need to add extra spacing between adjacent text elements -- the line height handles it.
- Avoid center-aligning paragraphs and smaller body text. Center alignment works for short headings or single lines, but it makes longer text harder to read. Left-align body copy.

---

## 5. Color

Use color with purpose, not decoration.

- Start with one primary brand color. Lighten it for backgrounds, darken it for text -- this adds cohesion without adding complexity.
- Semantic colors have universal meaning. Respect them:
  - Blue: trust, links, primary actions
  - Red: danger, errors, urgency
  - Yellow: warnings
  - Green: success, positive states
- Build a color ramp from the primary: tints and shades of one hue give you chips, states, and charts without visual chaos.
- Let color emerge from function first (focus states, error states, status chips) rather than applying it decoratively.

---

## 6. Dark Mode

Dark mode has different rules than light mode.

- In light mode, shadows create depth. In dark mode, shadows disappear -- use surface elevation (lighter cards than background) instead.
- Borders in dark mode should have low contrast. High-contrast light borders on dark backgrounds look harsh.
- Bright chips and badges need their saturation reduced in dark mode, with high-contrast text on top.
- Dark mode is not just navy blue and gray -- deep purples, reds, and greens all work.

---

## 7. Shadows

Shadows should be felt, not seen.

- Default tool shadows (Figma's included) are almost always too strong. Reduce opacity and increase blur.
- Change the shadow color from black to a muted gray for a softer result.
- Cards need subtle shadows. Popovers and overlays need stronger ones (more visual separation = stronger shadow).
- If the shadow is the first thing you notice in a design, it is wrong.
- Inner and outer shadows can simulate tactile raised buttons -- use sparingly.
- Shadows can replace solid borders. A soft shadow around a card creates separation without drawing a hard line. Often looks cleaner than a visible border.
- Perceived depth attracts focus. Elements that feel closer to the user draw the eye first. Use elevation (shadow, z-index, surface color) intentionally to pull attention to what matters.

---

## 8. Icons

Good icons speed up scanning. Bad or mismatched icons slow it down.

- Match icon size to the line height of adjacent text (e.g. if body text line height is 24px, use 24px icons).
- Use a single icon library throughout a component area. Mixing stroke weights, fill styles, or corner treatments looks inconsistent.
- Exception: different areas of the UI (nav, content cards, category chips) can use different icon styles if they are visually separated and used for different purposes.
- Universal icons (home, bookmark, user, search) need no labels. Ambiguous or unfamiliar icons need a label or tooltip.
- Avoid decorative/bizarre icons. Icons should reduce cognitive load, not add to it.

---

## 9. Buttons

Buttons have consistent rules that apply everywhere.

- Every button needs at minimum 4 states: default, hover, active/pressed, disabled.
- Add a loading state (spinner) for async actions.
- Button padding rule of thumb: horizontal padding should be roughly double the vertical padding.
- Ghost buttons (no background until hover) are good for secondary CTAs next to a primary button.
- Sidebar nav links are just ghost buttons without a background -- treat them the same way.

---

## 10. Form Inputs

Inputs are higher-stakes than buttons. More states required.

- Focus state: visible ring or border change when the user clicks in
- Error state: red border + error message below the field
- Warning state: for optional/advisory issues
- Never let an input silently fail. Every state needs a visible response.

---

## 11. Interactive Feedback

Every action needs a response. Silence feels broken.

- Loading spinner while data fetches
- Success message when an action completes
- Disabled/grayed-out state immediately when a button is clicked (before the next screen loads) so the user knows their click registered
- Fill-in icons, badge updates, and other state changes as confirmation of actions
- Micro interactions (e.g. a chip sliding up to confirm a copy action) are a tier above basic feedback -- they confirm the outcome, not just the interaction

---

## 12. Micro Interactions

Micro interactions confirm outcomes and add character.

- Basic feedback: hover state changes, click state changes
- Micro interaction: a secondary animated element confirms what happened (text "Copied!" sliding up, a badge count incrementing, a fill animating on save)
- Range from purely functional to playful -- both are valid depending on the product tone
- Do not add these for decoration. They should always communicate something.

---

## 13. Image Overlays

When text sits over an image, you need a proper overlay.

- A flat full-screen color overlay kills the image. Avoid it.
- Use a linear gradient that fades from transparent (showing the image) into a solid readable background color.
- For a more modern effect, add a progressive blur on top of the gradient.

---

## 14. User Flow Planning

Plan the flow before designing screens. Missing states are invisible to designers but immediately felt by users.

- Sketch boxes on paper or whiteboard before going to pixels. Catch gaps early.
- Common missed items: skip/back buttons, empty states, edge case inputs (e.g. an allergy not in the preset list), navigation for every direction a user might go
- Check: can every screen be exited? Is there a way forward for every user scenario?
- Small completions matter: a filter icon on a search bar, a save button in the header, a "clear all" option.

---

## 15. Consistency

Inconsistency makes designs look amateur. It is also usually easy to fix.

- Same component used in two places should be identical except for its content.
- Use a shared token/variable for corner radius, spacing, and color values. Changing one value updates everything.
- Same-purpose components (two search bars, two back buttons) must match in size, corner radius, and style.
- Smaller components (chips, tags, pills) typically use a smaller corner radius (around 10px) than large containers.

---

## 16. Redundant Elements

When something does not add information, remove it.

- Decorative arrows on swipeable carousels (swipe affordance handles it)
- Strokes/borders on elements that already have sufficient contrast via background color
- Repeated labels or icons for actions that are already obvious
- Ask: does this element teach the user something, or is it just visual habit?

---

## 17. Charts and Data Visualization

Over-designed charts obscure data. Function beats aesthetics here.

- Always include a readable vertical axis.
- Flat-top bars are easier to read precisely than rounded tops.
- Data points must map to actual data. Do not pad or stylize the count of items.
- A simpler, less aesthetic chart that conveys accurate information is better than a beautiful chart that is hard to read.
- Dribbble-style charts are portfolio work. Production charts should prioritize legibility.

---

## 18. Start Simple: Design the Minimum First

Start from the core functionality of a page, not its structure.

- Ask: what is the one thing this page needs to do? For most websites that is a heading, an input, and a button. Start there.
- Resist the urge to design top-down from nav to footer. Design the essential element first, then build outward only as needed.
- More elements almost always means a worse design, not a better one. Add only what earns its place.
- Do not design with placeholder text or lorem ipsum. Spacing that works for one piece of real content may break for another. Use actual content early.

---

## 19. Gestalt Laws: Grouping and Scannability

Users process a design as a whole before reading any individual part. Design for that first pass.

- The design should be scannable and understandable in a few seconds. If it is not, reduce complexity.
- Law of proximity: elements placed close together are perceived as a group. Use tighter spacing within a group and larger gaps between groups to communicate structure without labels.
- Law of similarity: elements that share shape, size, or color are perceived as related. Use this to show that a set of items belong together.
- Your first goal is always: does the layout make sense at a glance?

---

## 20. Depth and Visual Interest

Flat designs can feel lifeless. A few targeted techniques add polish without clutter.

- Replace a solid color button or badge with a subtle single-hue gradient. Going from one shade of a color to a slightly lighter or darker shade of the same color adds depth without looking garish.
- Use accent colors to highlight the single most important interactive element on the screen.
- Cards can make bland list content feel more structured and scannable.
- The goal is character, not decoration. Each depth technique should serve the content, not compete with it.

---

## 21. The Creative Process

Creativity is a process, not a flash of inspiration. It can be structured.

1. Know the fundamentals. Rules give you a base to work from and bend intentionally.
2. Gather inspiration before designing. Study top-tier websites and apps. Save the ones you like. Do not try to design from a blank slate.
3. Analyze what you saved. Ask why you like it. What is simple? What is unique? What would you borrow? Note two or three concrete ideas.
4. Step away. Once you have initial ideas, do not act on them immediately. Take a break. Let the ideas develop passively. When you return, new approaches will surface.
5. Design something. Anything. The first version does not need to be good. The goal is to produce something you can react to and improve.
6. Do not fall in love with your work. Show it to colleagues. Test it with users. Adjust based on what you learn. A good pricing section discovered in your third failed design attempt is still a win.

---

## Quick Reference Checklist

Before calling a design done:

- [ ] Every interactive element has hover, active, disabled states
- [ ] Every form input has focus, error, and warning states
- [ ] Loading and success states exist for async actions
- [ ] Visual hierarchy is clear: most important thing stands out when you zoom out and scan for 2 seconds
- [ ] Secondary content is de-emphasized, not just primary content emphasized
- [ ] Spacing values come from the scale: 4 8 12 16 20 28 40 60 100 160 240
- [ ] One font family, max 6 sizes for landing pages
- [ ] Body and paragraph text is left-aligned, not centered
- [ ] Smaller text has larger line height than large headings
- [ ] Icons are from the same library within each area
- [ ] Colors are used for meaning, not decoration
- [ ] No redundant or decorative elements that add noise
- [ ] User can always move forward and backward in any flow
- [ ] Image overlays use gradients, not flat color blocks
- [ ] Shadows are subtle -- felt not seen
- [ ] Design started from the core functionality, not the layout shell
