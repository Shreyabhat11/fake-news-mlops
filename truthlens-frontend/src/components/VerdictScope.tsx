/**
 * A static, hand-built illustration standing in for the hero's "characteristic
 * moment": a dial reading between REAL and FAKE, with a waveform beneath it
 * evoking the embedding/semantic-analysis step. Deliberately restrained —
 * two semantic accent colors only, no gratuitous gradients.
 */
export function VerdictScope() {
  return (
    <svg
      viewBox="0 0 420 420"
      className="h-full w-full"
      role="img"
      aria-label="Illustration of a credibility dial reading toward real, with an analysis waveform beneath it"
    >
      <defs>
        <linearGradient id="arcReal" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#3DDC97" stopOpacity="0.35" />
          <stop offset="100%" stopColor="#3DDC97" stopOpacity="0.95" />
        </linearGradient>
        <linearGradient id="arcFake" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#F2665A" stopOpacity="0.95" />
          <stop offset="100%" stopColor="#F2665A" stopOpacity="0.35" />
        </linearGradient>
        <radialGradient id="hubGlow" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#3DDC97" stopOpacity="0.5" />
          <stop offset="100%" stopColor="#3DDC97" stopOpacity="0" />
        </radialGradient>
      </defs>

      {/* outer static ring */}
      <circle cx="210" cy="190" r="150" fill="none" stroke="rgba(232,234,237,0.06)" strokeWidth="1" />
      <circle cx="210" cy="190" r="128" fill="none" stroke="rgba(232,234,237,0.05)" strokeWidth="1" />

      {/* gauge arc: fake (left) to real (right), 180 degrees */}
      <path
        d="M 80 190 A 130 130 0 0 1 210 60"
        fill="none"
        stroke="url(#arcFake)"
        strokeWidth="10"
        strokeLinecap="round"
      />
      <path
        d="M 210 60 A 130 130 0 0 1 340 190"
        fill="none"
        stroke="url(#arcReal)"
        strokeWidth="10"
        strokeLinecap="round"
      />

      {/* tick marks */}
      {Array.from({ length: 9 }).map((_, i) => {
        const angle = Math.PI - (i * Math.PI) / 8
        const x1 = 210 + Math.cos(angle) * 118
        const y1 = 190 - Math.sin(angle) * 118
        const x2 = 210 + Math.cos(angle) * 132
        const y2 = 190 - Math.sin(angle) * 132
        return (
          <line
            key={i}
            x1={x1}
            y1={y1}
            x2={x2}
            y2={y2}
            stroke="rgba(232,234,237,0.18)"
            strokeWidth="1.5"
          />
        )
      })}

      {/* needle pointing toward "real" at ~35 degrees past center, reflecting a confident-but-not-absolute reading */}
      <g transform="rotate(-52 210 190)">
        <line x1="210" y1="190" x2="210" y2="78" stroke="#E8EAED" strokeWidth="2.5" strokeLinecap="round" />
        <polygon points="210,72 205,86 215,86" fill="#E8EAED" />
      </g>
      <circle cx="210" cy="190" r="34" fill="url(#hubGlow)" />
      <circle cx="210" cy="190" r="9" fill="#0A0E14" stroke="#3DDC97" strokeWidth="2" />

      <text x="86" y="222" fontSize="11" fill="#F2665A" fontFamily="JetBrains Mono, monospace" letterSpacing="0.5">
        FAKE
      </text>
      <text x="316" y="222" fontSize="11" fill="#3DDC97" fontFamily="JetBrains Mono, monospace" letterSpacing="0.5">
        REAL
      </text>

      {/* semantic embedding waveform beneath the dial */}
      <g transform="translate(55, 320)" opacity="0.85">
        <polyline
          points="0,20 20,20 30,4 40,34 50,10 60,26 70,20 90,20 100,2 110,38 120,16 130,24 140,20 160,20 170,8 180,30 190,14 200,22 210,20 230,20 240,6 250,32 260,18 270,20 290,20 300,10 310,28 320,20"
          fill="none"
          stroke="#7C8CF8"
          strokeWidth="1.75"
          strokeLinecap="round"
          strokeLinejoin="round"
          opacity="0.55"
        />
      </g>
    </svg>
  )
}
