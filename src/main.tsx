import ReactDOM from 'react-dom/client'

import { BubbleAnimation } from './Animation'

const root = ReactDOM.createRoot(document.getElementById('root') as HTMLElement)

root.render(<BubbleAnimation state="idle" frequencies={[]} />)
