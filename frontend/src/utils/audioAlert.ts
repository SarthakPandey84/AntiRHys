/**
 * AntiRHys Web Audio Alert Service
 *
 * Provides browser-side audio alerts using Web Audio API:
 * - Handles browser autoplay policies and AudioContext resumption
 * - Generates alternating siren tone without external audio files
 * - Includes an explicit test sound routine for driver pre-checks
 * - Ensures single active timer/oscillator loop without resource leaks
 */

export type AudioStatus = 'untested' | 'ready' | 'blocked' | 'muted';

class AudioAlertService {
  private ctx: AudioContext | null = null;
  private intervalId: number | null = null;
  private isHighTone = false;
  private status: AudioStatus = 'untested';
  private isMuted = false;
  private listeners: ((status: AudioStatus) => void)[] = [];

  constructor() {
    // Lazy initialization
  }

  public subscribe(listener: (status: AudioStatus) => void): () => void {
    this.listeners.push(listener);
    listener(this.status);
    return () => {
      this.listeners = this.listeners.filter((l) => l !== listener);
    };
  }

  private notify() {
    this.listeners.forEach((listener) => listener(this.status));
  }

  public getStatus(): AudioStatus {
    return this.isMuted ? 'muted' : this.status;
  }

  public setMuted(muted: boolean) {
    this.isMuted = muted;
    if (muted) {
      this.stopAlarm();
    }
    this.notify();
  }

  public async initAudio(): Promise<boolean> {
    try {
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      if (!AudioCtx) {
        this.status = 'blocked';
        this.notify();
        return false;
      }

      if (!this.ctx) {
        this.ctx = new AudioCtx();
      }

      if (this.ctx.state === 'suspended') {
        await this.ctx.resume();
      }

      if (this.ctx.state === 'running') {
        if (this.status !== 'ready') {
          this.status = 'ready';
          this.notify();
        }
        return true;
      } else {
        this.status = 'blocked';
        this.notify();
        return false;
      }
    } catch (err) {
      console.warn('AudioContext initialization failed:', err);
      this.status = 'blocked';
      this.notify();
      return false;
    }
  }

  /**
   * Play a brief, non-intrusive test chime to verify speaker audio and unlock browser autoplay.
   */
  public async playTestSound(): Promise<boolean> {
    const success = await this.initAudio();
    if (!success || !this.ctx) return false;

    try {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();

      osc.type = 'sine';
      osc.frequency.setValueAtTime(600, this.ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(880, this.ctx.currentTime + 0.25);

      gain.gain.setValueAtTime(0.2, this.ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.35);

      osc.connect(gain);
      gain.connect(this.ctx.destination);

      osc.start();
      osc.stop(this.ctx.currentTime + 0.35);

      this.status = 'ready';
      this.notify();
      return true;
    } catch (err) {
      console.error('Failed to play test sound:', err);
      this.status = 'blocked';
      this.notify();
      return false;
    }
  }

  private playToneStep() {
    if (!this.ctx || this.isMuted || this.ctx.state !== 'running') return;

    try {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();

      osc.type = 'sawtooth';
      const freq = this.isHighTone ? 950 : 750;
      this.isHighTone = !this.isHighTone;

      osc.frequency.setValueAtTime(freq, this.ctx.currentTime);

      // Volume envelope to avoid clicks
      gain.gain.setValueAtTime(0.5, this.ctx.currentTime);
      gain.gain.setValueAtTime(0.5, this.ctx.currentTime + 0.4);
      gain.gain.exponentialRampToValueAtTime(0.0001, this.ctx.currentTime + 0.48);

      osc.connect(gain);
      gain.connect(this.ctx.destination);

      osc.start();
      osc.stop(this.ctx.currentTime + 0.48);
    } catch (e) {
      console.warn('Audio playback step error:', e);
    }
  }

  private shouldBePlaying = false;
  private isStarting = false;

  public startAlarm() {
    if (this.isMuted) return;
    this.shouldBePlaying = true;
    if (this.intervalId !== null || this.isStarting) return;

    this.isStarting = true;
    this.initAudio()
      .then((ready) => {
        this.isStarting = false;
        if (!ready) return;
        // If stopAlarm was invoked while initAudio was resolving, abort scheduling
        if (!this.shouldBePlaying || this.isMuted) return;
        if (this.intervalId !== null) return;
        this.isHighTone = false;
        this.playToneStep();
        this.intervalId = window.setInterval(() => this.playToneStep(), 500);
      })
      .catch((err) => {
        this.isStarting = false;
        console.warn('Audio alarm initialization failed:', err);
      });
  }

  public stopAlarm() {
    this.shouldBePlaying = false;
    this.isStarting = false;
    if (this.intervalId !== null) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
  }

  public cleanup() {
    this.stopAlarm();
    if (this.ctx && this.ctx.state !== 'closed') {
      try {
        this.ctx.close();
      } catch {
        // Ignore close errors
      }
      this.ctx = null;
    }
    this.status = 'untested';
    this.notify();
  }
}

export const audioAlertService = new AudioAlertService();
