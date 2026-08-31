import { AlertTriangle, Camera, CheckCircle2, Copy, LoaderCircle } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { KernelLanguage, SavedVersion } from '../domain/types'
import { sourceHash } from '../lib/hash'
import { languageLabel } from '../lib/languages'
import { Modal } from './Modal'

export interface SaveVersionPayload {
  name: string
  notes: string
  source: string
  language: KernelLanguage
  allowDuplicate: boolean
}

export function SaveVersionDialog({
  open,
  problemId,
  language,
  snapshot,
  currentSource,
  existingVersions,
  busy,
  onClose,
  onSave,
}: {
  open: boolean
  problemId: string
  language: KernelLanguage
  snapshot: string
  currentSource: string
  existingVersions: SavedVersion[]
  busy: boolean
  onClose: () => void
  onSave: (payload: SaveVersionPayload) => void
}) {
  const [name, setName] = useState('')
  const [notes, setNotes] = useState('')
  const [hash, setHash] = useState('')
  const [duplicates, setDuplicates] = useState<SavedVersion[]>([])
  const [checking, setChecking] = useState(false)
  const [allowDuplicate, setAllowDuplicate] = useState(false)

  useEffect(() => {
    if (!open) return
    setName('')
    setNotes('')
    setDuplicates([])
    setAllowDuplicate(false)
    setChecking(true)
    void sourceHash(snapshot).then(async (nextHash) => {
      setHash(nextHash)
      const local = existingVersions.filter((version) => version.language === language && version.source_hash === nextHash)
      try {
        const remote = await api.versions.findDuplicates(problemId, language, nextHash)
        const merged = [...local, ...remote].filter((version, index, all) => all.findIndex(({ id }) => id === version.id) === index)
        setDuplicates(merged)
      } catch {
        setDuplicates(local)
      } finally {
        setChecking(false)
      }
    })
  }, [open, problemId, language, snapshot, existingVersions])

  const isDuplicate = duplicates.length > 0
  const canSave = name.trim().length > 0 && !busy && !checking && (!isDuplicate || allowDuplicate)
  return (
    <Modal
      open={open}
      onClose={busy ? () => undefined : onClose}
      title="保存为性能版本"
      subtitle="平台会重新完整验证，再使用固定协议 benchmark；两步都成功后才会持久化。"
      footer={
        <>
          <button className="button ghost" type="button" onClick={onClose} disabled={busy}>取消</button>
          <button
            className="button primary"
            type="button"
            disabled={!canSave}
            onClick={() => onSave({ name: name.trim(), notes: notes.trim(), source: snapshot, language, allowDuplicate })}
          >
            {busy ? <LoaderCircle className="spin" size={16} /> : <CheckCircle2 size={16} />}
            {busy ? '正在提交…' : '验证、测速并保存'}
          </button>
        </>
      }
    >
      <div className="snapshot-banner">
        <Camera size={18} />
        <div>
          <strong>已锁定点击时的代码快照</strong>
          <span>{languageLabel(language)} · {snapshot.split('\n').length} 行 · SHA-256 {hash ? hash.slice(0, 12) : '计算中…'}</span>
        </div>
      </div>
      {currentSource !== snapshot && (
        <div className="inline-notice warning"><AlertTriangle size={16} />编辑器已有新修改；本次仍保存上面锁定的快照。</div>
      )}
      <label className="field">
        <span>版本名称 <b>*</b></span>
        <input autoFocus maxLength={80} value={name} onChange={(event) => setName(event.target.value)} placeholder="例如：共享内存分块 v2" />
        <small>{name.length}/80</small>
      </label>
      <label className="field">
        <span>备注 <em>可选</em></span>
        <textarea maxLength={500} rows={3} value={notes} onChange={(event) => setNotes(event.target.value)} placeholder="记录这次优化的思路、取舍或待验证假设…" />
        <small>{notes.length}/500</small>
      </label>
      {checking && <div className="duplicate-check"><LoaderCircle className="spin" size={15} />正在检查重复源码…</div>}
      {!checking && isDuplicate && (
        <div className="duplicate-warning" role="alert">
          <div><Copy size={17} /><strong>检测到相同源码</strong></div>
          <p>它已保存为「{duplicates[0].name}」。重复版本可能产生新的测量样本，你仍可以选择继续。</p>
          <label><input type="checkbox" checked={allowDuplicate} onChange={(event) => setAllowDuplicate(event.target.checked)} />我了解，仍要保存重复源码</label>
        </div>
      )}
    </Modal>
  )
}
