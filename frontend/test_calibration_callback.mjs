/**
 * Regression test for Frontend React command sender registration and state dispatch.
 *
 * Verifies:
 * 1. Bug reproduction: Calling setSendCommandFn(fn) directly in React causes React
 *    to evaluate fn(prevState). Since sendCommand(null) returns false, the state becomes false,
 *    causing onSendCommand to be undefined and clicks to be silently dropped.
 * 2. Bug fix: Calling setSendCommandFn(() => fn) properly stores the function reference,
 *    allowing onSendCommand to be invoked and WebSocket commands dispatched.
 */

import assert from 'node:assert';

console.log('--- Running Frontend Calibration Callback Registration Regression Test ---');

// Mock sendCommand callback as defined in CameraStream.tsx
let webSocketOpen = false;
let sentMessages = [];

const mockSendCommand = (command, extra = {}) => {
  if (!webSocketOpen) {
    return false;
  }
  sentMessages.push({ command, ...extra });
  return true;
};

// 1. Reproduce Bug: Standard React useState updater behavior
{
  let state = null;
  const setSendCommandFn = (val) => {
    // Exact React 18/19 useState dispatcher semantics:
    if (typeof val === 'function') {
      state = val(state);
    } else {
      state = val;
    }
  };

  // Buggy registration: passing setSendCommandFn directly
  // onRegisterSendCommand(sendCommand) -> setSendCommandFn(mockSendCommand)
  setSendCommandFn(mockSendCommand);

  // In the buggy implementation, state becomes false!
  assert.strictEqual(state, false, 'Expected buggy direct setter to evaluate function and store false');

  // Dashboard props resolution: onSendCommand={sendCommandFn || undefined}
  const onSendCommand = state || undefined;
  assert.strictEqual(onSendCommand, undefined, 'Expected onSendCommand to be undefined in buggy version');

  // Calling click handler with undefined onSendCommand does nothing
  let clicked = false;
  if (onSendCommand) {
    onSendCommand('start_calibration');
    clicked = true;
  }
  assert.strictEqual(clicked, false, 'Click was silently ignored because onSendCommand was undefined');
  console.log('✓ Verified bug reproduction: direct state setter evaluated callback and became false (onSendCommand = undefined)');
}

// 2. Verify Fix: Callback registration with wrapper (() => fn)
{
  let state = null;
  const setSendCommandFn = (val) => {
    if (typeof val === 'function') {
      state = val(state);
    } else {
      state = val;
    }
  };

  const handleRegisterSendCommand = (fn) => {
    setSendCommandFn(() => fn);
  };

  // Fixed registration:
  handleRegisterSendCommand(mockSendCommand);

  // State is now the actual function!
  assert.strictEqual(typeof state, 'function', 'Expected fixed setter to store function');
  assert.strictEqual(state, mockSendCommand, 'Expected stored function to match mockSendCommand');

  const onSendCommand = state || undefined;
  assert.strictEqual(typeof onSendCommand, 'function');

  // Verify command dispatching when WebSocket is connected
  webSocketOpen = true;
  sentMessages = [];
  const result = onSendCommand('start_calibration', { driver_id: 'driver' });

  assert.strictEqual(result, true);
  assert.strictEqual(sentMessages.length, 1);
  assert.strictEqual(sentMessages[0].command, 'start_calibration');
  assert.strictEqual(sentMessages[0].driver_id, 'driver');
  console.log('✓ Verified bug fix: handleRegisterSendCommand preserves callback function and dispatches command');
}

console.log('--- All frontend calibration callback tests passed successfully! ---');
