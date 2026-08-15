import { Fragment, type ReactNode } from 'react'

function inline(text: string): ReactNode[] {
  return text.split(/(`[^`]+`|\*\*[^*]+\*\*)/g).filter(Boolean).map((part, index) => {
    if (part.startsWith('`') && part.endsWith('`')) return <code key={index}>{part.slice(1, -1)}</code>
    if (part.startsWith('**') && part.endsWith('**')) return <strong key={index}>{part.slice(2, -2)}</strong>
    return <Fragment key={index}>{part}</Fragment>
  })
}

export function MarkdownText({ source }: { source: string }) {
  const lines = source.replace(/\r\n/g, '\n').split('\n')
  const nodes: ReactNode[] = []
  let index = 0
  while (index < lines.length) {
    const line = lines[index]
    if (line.startsWith('```')) {
      const language = line.slice(3).trim()
      const code: string[] = []
      index += 1
      while (index < lines.length && !lines[index].startsWith('```')) code.push(lines[index++])
      nodes.push(<pre className="statement-code" key={`code-${index}`}><code data-language={language}>{code.join('\n')}</code></pre>)
    } else if (/^#{1,3} /.test(line)) {
      const level = line.match(/^#+/)?.[0].length ?? 2
      const content = line.replace(/^#{1,3}\s+/, '')
      nodes.push(level === 1 ? <h1 key={index}>{inline(content)}</h1> : level === 2 ? <h2 key={index}>{inline(content)}</h2> : <h3 key={index}>{inline(content)}</h3>)
    } else if (/^[-*] /.test(line)) {
      const items: string[] = []
      while (index < lines.length && /^[-*] /.test(lines[index])) items.push(lines[index++].slice(2))
      nodes.push(<ul key={`list-${index}`}>{items.map((item, itemIndex) => <li key={itemIndex}>{inline(item)}</li>)}</ul>)
      continue
    } else if (/^\d+\. /.test(line)) {
      const items: string[] = []
      while (index < lines.length && /^\d+\. /.test(lines[index])) items.push(lines[index++].replace(/^\d+\. /, ''))
      nodes.push(<ol key={`ordered-${index}`}>{items.map((item, itemIndex) => <li key={itemIndex}>{inline(item)}</li>)}</ol>)
      continue
    } else if (line.startsWith('> ')) {
      nodes.push(<blockquote key={index}>{inline(line.slice(2))}</blockquote>)
    } else if (line.trim()) {
      nodes.push(<p key={index}>{inline(line)}</p>)
    }
    index += 1
  }
  return <div className="markdown-text">{nodes}</div>
}
