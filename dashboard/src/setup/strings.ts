// Every user-facing sentence of the setup guide, the Wren panel and the reset section (D11): plain
// English for people who have never run a model server, short, no jargon without a number next to
// it. `~/writing/check.sh` runs on an extract of this file (tests/setup-strings.test.ts writes it).
// Labels that come from the registry (component names, field labels, the "what" sentences) are the
// registry's own text and live in models/<id>/model.yaml.

import type { SetupJobStepState, SetupStepId } from '../api/types';

const gib = (n: number) => `${n.toFixed(1)} GiB`;
const tokens = (n: number) => Math.round(n).toLocaleString('en-US');

export const STEP_LABEL: Record<SetupStepId, string> = {
  models: 'Models',
  overview: 'Model',
  weights: 'Weights',
  components: 'Tools',
  parallelism: 'Memory',
  methods: 'Methods',
  review: 'Review',
  install: 'Install',
};

export const STEP_TITLE: Record<SetupStepId, string> = {
  models: 'Pick a model',
  overview: 'What you get',
  weights: 'Choose the weights',
  components: 'Tools that run beside the model',
  parallelism: 'Parallel chats and memory',
  methods: 'Methods and advanced settings',
  review: 'Check before you install',
  install: 'Installing',
};

export const STEP_INTRO: Record<SetupStepId, string> = {
  models: 'Each tile is a model this box can serve. Pick one to see how fast it runs here and to set it up.',
  overview: 'What the model is, and how fast each size of it runs on a DGX Spark. Sort the table or switch the mode to compare the sizes.',
  weights: 'Smaller weights run faster and use less memory, but their answers drift further from the original model. KL measures that drift: lower is closer.',
  components: 'We checked what we run on our own box. Each tool says what it does and what it costs in memory.',
  parallelism: 'Each slot is one chat the model can serve at the same time. The window is the longest conversation a slot can hold.',
  methods: 'The recommended values are already set. Change one only if you know why.',
  review: 'Nothing has changed on the box yet. This is what Finish will do.',
  install: 'The box runs the steps below. You can leave this page; it picks up where it was when you come back.',
};

/** One line per step for the tiles page's "what the guide walks you through" list. */
export const STEP_SHORT: Record<SetupStepId, string> = {
  models: 'Pick a model',
  overview: 'See how fast each size runs on this box',
  weights: 'Pick a size that fits your memory and the quality you want',
  components: 'Turn the helper tools on or off',
  parallelism: 'Decide how many chats run at once and how long they can be',
  methods: 'Keep the recommended settings, or change one',
  review: 'See every change before anything happens',
  install: 'The box fetches what it needs, then restarts with your settings',
};

export const S = {
  guideSteps: 'What the guide walks you through',
  pageTitle: 'Setup guide',
  stepOf: (n: number, total: number) => `Step ${n} of ${total}`,
  next: 'Next',
  previous: 'Previous',
  finish: 'Finish and install',
  recommended: 'Recommended',
  recommendedValue: 'recommended',
  resetToRecommended: 'Use the recommended values',
  loading: 'Loading the setup guide',
  retry: 'Try again',
  storageOff: 'This browser does not let the page save, so your progress will not be kept.',
  storageCorrupt: 'The guide could not read the saved progress, so it starts from the recommended setup.',
  storageOld: 'The saved progress is from an older version of the guide, so it starts from the recommended setup.',
  savedProfileGone: (id: string) => `The guide no longer offers the saved weights ${id}, so it picked the recommended ones.`,
  savedComponentGone: (id: string) => `The guide no longer offers the saved tool ${id} and left it out.`,
  savedValueBad: (label: string) => `The saved value of ${label} is no longer valid, so the guide went back to the recommended one.`,
  modelsEmpty: 'The box lists no models: its registry folder models/ is empty.',
  modelsError: 'The box did not answer with its list of models.',
  modelLoadErrors: (n: number) => (n === 1 ? 'The box could not read one model file:' : `The box could not read ${n} model files:`),
  unauthorized: 'This box has an admin password, and the dashboard has no sign-in yet. Unset QSE_ADMIN_TOKEN to use the setup guide (docs/server.md).',
  setupModeTab: 'Not available while the box is in setup mode',
  setupModeBanner: 'The box is in setup mode: only the dashboard and Wren are running. Finish the setup to start the model.',
  resetLink: 'Reset',
  resetHeading: 'Reset',

  // tiles
  tileParams: (p: string) => `${p} parameters`,
  tileBest: (tok: number, task: string, profile: string) => `${tok.toFixed(1)} tok/s on ${task} with ${profile}`,
  tileOpen: 'Set up',
  tileRecommended: 'Recommended',
  deviceLine: (total: number, free: number | null) => (free == null ? `This box has ${gib(total)} of memory.` : `This box has ${gib(total)} of memory, ${gib(free)} free right now.`),

  // overview
  license: 'License',
  measuredOn: (build: string, date: string) => `Measured with build ${build} on ${date}, one request at a time, with the caches off.`,
  howMeasured: 'How we measured',
  modeGreedy: 'Greedy',
  modeSampled: 'Sampled',
  tokens512: '512 tokens',
  tokens128: '128 tokens',
  groupEveryday: 'Everyday',
  groupLong: 'Long context',
  colWeights: 'Weights',
  colProse: 'Prose',
  colChat: 'Chat',
  colCode: 'Code',
  colLong32: 'After 32k',
  colLong64: 'After 64k',
  colTtft8: 'First token at 8k',
  colTtft32: 'First token at 32k',
  colKl: 'KL',
  colLossless: 'Same text',
  tokPerS: 'tok/s',
  seconds: 's',
  notComparable: 'older build, not comparable',
  selectedRow: 'your choice',
  losslessTitle: (same: number, tie: number) => `${same} of ${same + tie} traces identical to the greedy decode, ${tie} differ only at a near tie`,
  sortBy: (col: string) => `Sort by ${col}`,
  chooseWeights: 'Choose the weights',

  // weights
  weightsKl: 'KL vs original',
  weightsTop1: 'Same top token',
  weightsSpeed: 'Prose and code',
  weightsMemory: 'Memory',
  weightsDisk: 'Download',
  onDisk: 'On this box',
  toDownload: (gb: number) => `${gb.toFixed(1)} GB`,
  privateRepo: 'The EXL3 weights are in private repositories. This box has a Hugging Face token, so it can download them.',
  privateNoToken: 'This box has no Hugging Face token, so it cannot download these weights. Add one with huggingface-cli login on the box, or pick weights that are already here.',
  activeNow: 'Running now',
  privateNoTokenShort: 'Needs a Hugging Face token, and this box has none.',
  notMeasured: 'not measured on this build',

  // components
  groups: { speed: 'Speed', roles: 'Helper models', caches: 'Caches', disk: 'Disk', features: 'Features' } as Record<string, string>,
  alwaysOn: 'Always on',
  needs: (labels: string[]) => `Needs ${joinAnd(labels)}`,
  conflictsWith: (labels: string[]) => `Cannot run with ${joinAnd(labels)}`,
  cascadeOff: (cause: string, labels: string[]) => `${joinAnd(labels)} ${labels.length === 1 ? 'needs' : 'need'} ${cause}, so ${labels.length === 1 ? 'it is' : 'they are'} off too.`,
  memoryCost: (n: number) => (n > 0 ? `${gib(n)}` : 'no GPU memory'),
  memoryUpTo: (n: number) => `up to ${gib(n)}`,
  diskCost: (n: number) => `${n.toFixed(n < 1 ? 1 : 0)} GB on disk`,
  learnMore: 'Docs',

  // parallelism
  slots: 'Parallel chats',
  slotsWhat: 'How many chats the model serves at the same time. Each one shares the GPU, so each runs slower when all are busy.',
  window: 'Longest conversation',
  windowWhat: 'The most tokens one chat can hold. Memory is only used as a chat grows, so a long window costs nothing until it is used.',
  sideWindow: 'Longest conversation in the extra chats',
  sideWindowWhat: 'The same limit for chats 2 to 4. Lower it to keep room for one long session.',
  kvPool: 'Shared memory pool',
  kvPoolWhat: 'Chats take memory from one pool as they grow. Off, every chat holds its whole window from the start.',
  floor: 'Memory kept free',
  floorWhat: 'The engine stops taking new memory when less than this is free, so the box stays responsive.',
  windowClamped: (n: number) => `Set to ${tokens(n)} tokens, the nearest length this model supports.`,
  fitsUnit: 'tokens of conversation fit, over all chats.',
  fitsPerSlot: (n: number, slots: number) => (slots > 1 ? `That is about ${tokens(n)} tokens for each of ${slots} chats if all are long at once.` : `One chat can grow to about ${tokens(n)} tokens.`),
  worstCase: (gb: number, fits: boolean) => (fits ? `Even with every chat at its full window (${gib(gb)}), it fits.` : `Every chat at its full window would need ${gib(gb)}. The engine pauses or parks a chat before that happens.`),
  tokensUnit: 'tokens',

  // methods
  advanced: 'Advanced',
  advancedNote: 'Where the engine listens, and the admin password.',
  invalidValue: (why: string) => `Not valid: ${why}.`,

  // review
  reviewChanges: 'What changes',
  reviewNoChanges: 'Nothing in serve.env changes.',
  reviewKey: 'Setting',
  reviewNow: 'Now',
  reviewAfter: 'After',
  reviewFrom: 'Set by',
  reviewUnset: 'unset',
  reviewWeights: 'Weights',
  reviewWeightsSame: (p: string) => `${p}, already running.`,
  reviewWeightsSwitch: (from: string | null, to: string) => (from ? `${from} to ${to}.` : `${to}.`),
  reviewDownload: (gb: number) => `${gb.toFixed(1)} GB to download first.`,
  reviewRestart: 'The engine restarts. Chats in progress stop, and the box answers again after about 4 minutes.',
  reviewNoRestart: 'No restart needed.',
  reviewCommands: 'What the box will run',
  reviewCommandsNote: 'For reading only. The box runs its own fixed steps; nothing typed on this page is ever run.',
  reviewMemory: 'Memory check',
  reviewFits: (total: number, budget: number) => `It fits: ${gib(total)} of the ${gib(budget)} the engine may use.`,
  reviewOver: (over: number) => `It does not fit: ${gib(over)} over. Finish stays off until it fits.`,
  howToFix: 'How to fix it',
  fixWindow: (n: number) => `Lower the longest conversation to ${tokens(n)} tokens`,
  fixSlots: (n: number) => `Use ${n} parallel ${n === 1 ? 'chat' : 'chats'}`,
  fixProfile: (label: string) => `Use the ${label} weights`,
  fixComponent: (label: string, gb: number) => `Turn off ${label} (${gib(gb)})`,
  fixPool: 'Turn on the shared memory pool',
  fixGo: 'Go there',
  overSentence: (over: number, hint: string | null) => `Over by ${gib(over)}.${hint ? ` ${hint}` : ''}`,
  overHint: (a: string, b: string | null) => (b ? `${a}, or ${lower(b)}.` : `${a}.`),
  planError: 'The box could not check this setup.',
  planLocal: 'The box did not answer, so this review comes from the page alone. Finish needs the box.',
  securityTitle: 'Anyone who can reach this port can change the setup',
  securityBody: 'This box has no admin password. Anyone on your network who opens this page can run the setup or reset the box. Set a password to stop that.',
  securityKeep: 'Keep it open',
  securityGenerate: 'Generate a password on the box',
  securityGenerateNote: 'The box writes it to secrets.env. Read it there with cat.',
  securityChoice: 'Admin password',
  securitySet: 'This box has an admin password.',
  confirmTitle: 'Install this setup?',
  confirmBody: (restart: boolean) => (restart ? 'Wren goes to sleep, the box writes the new settings and restarts the engine. Chats in progress stop.' : 'Wren goes to sleep and the box writes the new settings.'),
  confirmGo: 'Install',
  cancel: 'Cancel',
  blockedToken: 'Finish stays off: the weights need a Hugging Face token this box does not have.',
  blockedInvalid: (steps: string[]) => `Finish stays off: fix ${joinAnd(steps)} first.`,
  blockedBusy: 'Finish stays off while another setup job runs.',

  // install
  jobState: {
    queued: 'Waiting to start',
    running: 'Installing',
    done: 'Installed',
    failed: 'The install failed',
    rolled_back: 'The install failed and the box went back to the previous setup',
  } as Record<string, string>,
  stepState: { pending: 'waiting', running: 'running', done: 'done', failed: 'failed', skipped: 'skipped' } as Record<SetupJobStepState, string>,
  rolledBackTo: (p: string) => `The engine did not pass its check, so the box went back to ${p}. Nothing else changed.`,
  restarting: 'The engine is restarting. This takes about 4 minutes; the page keeps checking.',
  restartingSince: (s: number) => `Waiting for the engine for ${Math.round(s)} s.`,
  logTail: 'Log',
  openDashboard: 'See it run',
  openPlayground: 'Open the Playground',
  backToReview: 'Back to Review',
  installStarting: 'Starting the install',
  installError: 'The box did not start the install.',
  jobLost: 'The box does not know this install any more. It may have restarted without its job file.',

  // memory bar
  memTitle: 'Memory',
  memFits: (total: number, budget: number) => `${gib(total)} of ${gib(budget)}`,
  memLeft: (left: number) => `${gib(left)} left for chats`,
  memDevice: (total: number) => `of ${gib(total)} on the box`,
  memNow: (used: number) => `${gib(used)} in use now`,
  memFloor: 'kept free',
  memWorst: 'every chat at its full window',
  memLocal: 'Using a local estimate: the box did not answer.',
  memExpand: 'Show the memory details',
  memCollapse: 'Hide the memory details',
  memCat: { model: 'Model', helpers: 'Helper models', caches: 'Caches and features', engine: 'Engine', system: 'System', chats: 'Chats' } as Record<string, string>,

  // wren
  wrenName: 'Wren',
  wrenRole: 'Setup helper',
  wrenStatus: {
    idle: 'Ready',
    listening: 'Listening',
    thinking: 'Thinking',
    loading: 'Waking up',
    talking: 'Answering',
    acting: 'Changing a setting',
    error: 'Something went wrong',
    sleeping: 'Asleep',
  } as Record<string, string>,
  wrenAsleep: 'Wren is asleep.',
  wrenAsleepBody: 'Wren is a small model that runs on this box. It can explain each step and fill in the form for you.',
  wrenWake: 'Wake Wren',
  wrenWaking: (pct: number | null) => (pct == null ? 'Loading Wren' : `Loading Wren, ${Math.round(pct * 100)} %`),
  wrenCannotLoad: 'Wren cannot load right now.',
  wrenCannotLoadMemory: (need: number | null, free: number | null) =>
    need != null && free != null ? `Wren needs about ${gib(need)} and the box has ${gib(free)} to spare. The guide works without Wren.` : 'There is not enough free memory. The guide works without Wren.',
  wrenNoModel: 'Neither Qwen3.5-2B nor Qwen3.5-0.8B is on this box, so Wren cannot run. The guide works without it.',
  wrenFailed: 'Wren failed to load.',
  wrenFinished: 'Wren is asleep while the setup runs.',
  wrenBusy: 'Wren is answering in another tab.',
  wrenFallback: 'Running on the smaller Qwen3.5-0.8B, so answers are simpler.',
  wrenPlaceholder: 'Ask Wren about this step',
  wrenSend: 'Send',
  wrenStop: 'Stop',
  wrenSleep: 'Put Wren to sleep',
  wrenStopped: '(stopped)',
  wrenWentToSleep: 'Wren went to sleep.',
  wrenRetry: 'Try again',
  wrenError: 'Wren could not answer.',
  wrenOffline: 'The box did not answer. Check the connection and try again.',
  wrenCorrected: 'Wren tried something that does not exist and was corrected.',
  wrenRoundLimit: 'Wren stopped after 4 rounds of changes. Ask again if something is missing.',
  wrenUndo: 'Undo',
  wrenUndone: 'Undone',
  wrenSources: 'From',
  wrenOpen: 'Ask Wren',
  wrenClose: 'Close',
  wrenHello: 'Hi, I am Wren. Ask me what a setting does, or tell me what you want and I will set it.',
  wrenChips: {
    models: ['Which model should I pick?', 'What does this guide do?'],
    overview: ['What do these numbers mean?', 'What is greedy vs sampled?'],
    weights: ['Which size is best for my memory?', 'What is KL?'],
    components: ['What is the narrator?', 'Turn off vision'],
    parallelism: ['How many parallel chats do I need?', 'Use 2 slots'],
    methods: ['What does det mean?', 'What is the thinking effort?'],
    review: ['Is this setup safe?', 'Why does it restart?'],
    install: ['How long does this take?'],
  } as Record<SetupStepId, string[]>,
  toolDone: {
    toggleOn: (label: string) => `Wren turned on ${label}`,
    toggleOff: (label: string) => `Wren turned off ${label}`,
    profile: (label: string) => `Wren chose the ${label} weights`,
    field: (label: string, value: string) => `Wren set ${label} to ${value}`,
    step: (label: string) => `Wren opened ${label}`,
  },

  // tool errors (these go back to the model as the tool result)
  tool: {
    unknown: (name: string) => `Wren has no tool called ${name}. The tools are set_field, toggle, select_profile, go_to_step, next, previous, explain and read_device.`,
    userOnly: (name: string) => `${name} is for the user only. Ask the user to press it.`,
    badArgs: (why: string) => `The arguments are wrong: ${why}.`,
    noModel: 'No model is open yet. Ask the user to pick one on the Models step.',
    noStep: (step: string) => `The guide has no step ${step}. The steps are 0 models, 1 overview, 2 weights, 3 components, 4 parallelism, 5 methods, 6 review.`,
    noField: (step: string, field: string) => `The ${step} step has no field ${field}.`,
    noComponent: (id: string) => `This setup has no tool called ${id}.`,
    noProfile: (id: string) => `This model has no weights called ${id}.`,
    required: (label: string) => `${label} is always on; nobody can turn it off.`,
    secret: (label: string) => `${label} is a security setting. Only the user can change it.`,
    installStep: 'The install starts only when the user presses Finish.',
    atEnd: 'This is the last step before the install. Only the user can press Finish.',
    atStart: 'This is the first step already.',
    stale: 'The user changed this field after asking, so the user’s value stays.',
    ok: 'ok',
    same: 'It already had that value.',
  },
};

/** The reset section (VIS-82). */
export const R = {
  title: 'Reset',
  intro: 'Undo things in steps, from forgetting your setup progress to deleting the usage history. Each step shows what goes before the box deletes anything.',
  levelTitle: {
    1: 'Setup progress and Wren chat',
    2: 'Applied configuration',
    3: 'Caches and stored chats',
    4: 'Downloaded weights',
    5: 'Usage history and conductor data',
  } as Record<number, string>,
  levelWhat: {
    1: 'Forgets the choices in this browser and the chat with Wren. Nothing on the box changes.',
    2: 'Writes the shipped serve.env back and switches to the recommended weights.',
    3: 'Deletes the disk cache, parked sessions, the lookup store and the response cache.',
    4: 'Deletes every downloaded weight set except the one running now.',
    5: 'Deletes the usage ledger and the conductor’s data. The usage charts start empty.',
  } as Record<number, string>,
  restart: 'Restarts the engine',
  noRestart: 'No restart',
  browserOnly: 'This browser only',
  review: 'Review reset',
  danger: 'Cannot be undone',
  dialogItems: 'What goes',
  dialogTotal: (b: string) => `Total ${b}`,
  dialogNothing: 'This level has nothing to delete.',
  typeWord: (w: string) => `Type ${w} to confirm`,
  confirm: (n: number) => `Reset level ${n}`,
  confirmLocal: 'Forget my progress',
  cancel: 'Cancel',
  close: 'Close',
  working: 'Resetting',
  busy: (n: number | null) => (n == null ? 'Requests are running. Reset is available when they finish.' : `${n} ${n === 1 ? 'request is' : 'requests are'} running. Reset is available when they finish.`),
  done: (freed: string) => `Done. Freed ${freed}.`,
  doneLocal: 'Done. This browser forgot your progress and the chat.',
  partial: (n: number) => `The box could not delete ${n} ${n === 1 ? 'item' : 'items'}; the rest is gone.`,
  error: 'The box did not reset.',
  unauthorized: 'This box has an admin password, and the dashboard has no sign-in yet. Reset needs it (docs/server.md).',
  offline: 'The box did not answer. Check the connection and try again.',
  loadError: 'The box did not list its reset levels.',
  atLeast: (b: string) => `at least ${b}`,
  missing: 'not on the box',
  kept: (why: string) => `Stays: ${why}`,
  keptShort: 'stays',
  restartNeeded: 'Restart the engine to finish.',
  retry: 'Try again',
  openSetup: 'Open the setup guide',
};

export function stepRefLabel(id: SetupStepId): string {
  return STEP_TITLE[id];
}

export function joinAnd(xs: string[]): string {
  if (xs.length <= 1) return xs[0] ?? '';
  return `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`;
}

function lower(s: string): string {
  return s.charAt(0).toLowerCase() + s.slice(1);
}

/** Wren's own words for the sources it answered from. */
export function sourceLabel(path: string, title: string): string {
  return title || path;
}
