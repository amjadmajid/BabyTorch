-- Keep left-to-right code correct inside right-to-left (Arabic) text.
--
-- In an RTL document the bidi engine reverses code. The catch: a `\ttfamily`
-- switch (inline `\texttt` or a code block) inside Arabic makes polyglossia
-- look for an Arabic monospace font and abort the build. bidi's \LR / \mbox
-- do NOT avoid it; polyglossia's OWN english switch does. So we wrap code in
-- polyglossia's `english` language: \textenglish{} inline, the `english`
-- environment for blocks. That switches to English (LTR, Latin font) around
-- the code and back to Arabic after. Used only by the Arabic build.

function CodeBlock(el)
  return {
    pandoc.RawBlock('latex', '\\begin{english}'),
    el,
    pandoc.RawBlock('latex', '\\end{english}'),
  }
end

function Code(el)
  return {
    pandoc.RawInline('latex', '\\textenglish{'),
    el,
    pandoc.RawInline('latex', '}'),
  }
end
