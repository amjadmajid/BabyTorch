-- Keep left-to-right code correct inside right-to-left (Arabic) text.
--
-- In an RTL document the bidi engine reverses code blocks and reorders inline
-- code. We pin code left-to-right using polyglossia's own `english`
-- environment / \textenglish{} rather than the bidi package's \begin{LTR}:
-- under polyglossia + unicode-math, \begin{LTR} re-triggers the Arabic
-- font setup with the wrong current font and errors out, whereas the
-- polyglossia LTR does not. Used only by the Arabic build.

-- NOTE: block-level LTR for code is still being worked out. Both bidi's
-- \begin{LTR} and polyglossia's \begin{english} re-trigger polyglossia's
-- Arabic font setup with the wrong current font (a unicode-math interaction)
-- and abort the build, so CodeBlock wrapping is disabled for now. Inline code
-- (\LR) is fine. Consequence: fenced code blocks currently render right-to-left
-- reversed in the Arabic PDF -- a known issue to fix before shipping book-ar.

function Code(el)
  -- bidi's inline \LR is robust here (the block \begin{LTR} is what clashes
  -- with polyglossia, so blocks use the english environment above instead).
  return {
    pandoc.RawInline('latex', '\\LR{'),
    el,
    pandoc.RawInline('latex', '}'),
  }
end
