import './styles/tokens.css';
import './styles/app.css';
import './styles/playground.css';
import './charts/ribbon';
import './charts/bars';
import './charts/scatter';
import './charts/heatmap';
import './charts/timing-bar';
import './charts/spark';
import './views/usage';
import './views/live-panel';
import './views/performance';
import './views/dev';
import './views/system';
// views/playground (about 21 KB gzip: the chat, its Markdown, presets and export) loads on demand,
// the first time #/playground is opened (src/app.ts), so the pages watched on a phone stay small.
import './app';
