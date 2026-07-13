import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    rules: {
      'react-refresh/only-export-components': [
        'warn',
        { allowConstantExport: true },
      ],
      // React Compiler preview rule. Fires on legitimate "reset downstream
      // state when an upstream value changes" effects — pattern has no
      // concise non-effect replacement in React 19. Existing legitimate
      // sites carry a per-line disable with a reason; anything new fails
      // lint (--max-warnings 0), so the baseline can only shrink.
      'react-hooks/set-state-in-effect': 'warn',
    },
  },
  {
    // Files that legitimately mix components with config/registry exports:
    // shadcn ui/ follows upstream conventions (variants exports), router.tsx
    // is route wiring, PipelineSteps.tsx is the step registry (lazy panel
    // components + data table). HMR purity is not a concern for these.
    files: [
      'src/components/ui/**',
      'src/router.tsx',
      'src/components/episode/PipelineSteps.tsx',
    ],
    rules: {
      'react-refresh/only-export-components': 'off',
    },
  },
])
