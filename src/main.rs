extern crate num;
extern crate rustfft;

use hound;
use std::f32::consts::PI;
use std::i16;

// use rustfft::FFT;

use ndarray::Array2;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::FFTplanner;

fn write_wav() -> () {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create("sine.wav", spec).unwrap();
    for t in (0..44100).map(|x| x as f32 / 44100.0) {
        let sample = (t * 440.0 * 2.0 * PI).sin();
        let amplitude = i16::MAX as f32;
        writer.write_sample((sample * amplitude) as i16).unwrap();
    }
    writer.finalize().unwrap();
}

fn choppy(fname: &str) -> () {
    let mut reader = hound::WavReader::open(fname).expect("Failed to open WAV file");
    let spec = reader.spec();

    println!("Read rate {}", spec.sample_rate);
    println!(
        "Samples {} so time {}s",
        reader.len(),
        reader.len() / spec.sample_rate
    );
    let chopsize = (spec.sample_rate as f32 * 0.16) as u32;
    let stepsize = (spec.sample_rate as f32 * 0.04) as u32;

    let chopcount = (reader.len() / stepsize) - 3;

    // this is the width in Hz of each element in the FFT result
    // recall we pad the FFT with `chopsize` zeros
    let binwidth = spec.sample_rate / (chopsize * 2);

    let mut planner = FFTplanner::new(false);

    let mut signal = reader
        .samples::<i16>()
        .map(|x| Complex::new(x.unwrap() as f32, 0f32))
        .collect::<Vec<_>>();

    // make an overlapping grid of FFTs
    println!("chopcount {}", chopcount);
    let fft = planner.plan_fft((chopsize * 2) as usize);

    println!("spectra...");
    let spectra = (0..chopcount)
        .map(|c| {
            let offset = c * stepsize;

            let mut sig: Vec<Complex<f32>> = vec![Complex::zero(); chopsize as usize];
            let mut spectrum: Vec<Complex<f32>> = vec![Complex::zero(); (chopsize * 2) as usize];

            sig.copy_from_slice(&signal[offset as usize..(offset + chopsize) as usize]);
            let mut zeros: Vec<Complex<f32>> = vec![Complex::zero(); chopsize as usize];

            sig.extend_from_slice(&zeros);

            fft.process(&mut sig, &mut spectrum[..]);

            spectrum
        })
        .collect::<Vec<_>>();

    // debug noise, get rid of this
    // confirm expected peak signal in each chop
    for s in &spectra {
        let max_peak = s
            .iter()
            .take(chopsize as usize)
            .enumerate()
            .max_by_key(|&(_, freq)| (freq.norm() as u32))
            .map(|(b, _)| b * binwidth as usize);
        // println!("Peak at {:?}", max_peak);
    }
    // println!("We looked at {} candidate frequencies", chopsize);
    // end debug noise

    // these will be our starting frequencies for searching.
    // since each there are 6 costas freq and each is 6.25Hz above the previous,
    // we want to end on (max freq in our passband) - (6 * 6.25)
    let top_of_passband = 2500.0;
    let last_step = top_of_passband - (6.0 * 6.25);

    // number of base frequencies we're going to consider
    let steps = (last_step / 3.125) as u32;

    println!("searching...");

    let mut res = Vec::new();

    // these are frequency steps
    println!("Frequency steps {}", steps);

    // spec says 2 seconds before and 3 seconds after, but we don't have that much data!
    // we have exactly 15 seconds of data. message is 12.94 seconds. so we only have an extra 2.06 seconds...
    let start_times = ((1.0 / 0.04) + (1.0 / 0.04)) as usize;
    println!("Start times (time steps): {}", start_times);

    // do the search

    let mut result = Array2::zeros((steps as usize, start_times));

    // step through base frequencies
    for step in 0..steps {
        let base_fq = 3.125 + (step as f32 * 3.125);

        // these are time steps
        for base_time in 0..start_times {
            // base_fq is the 0 freq for our costas array

            // great, we're going to assume there's a costa array
            // based at (base_time, base_fq). let's take the energy of it.

            let mut sum = 0.0;
            let mut normal = 0.0;

            for sync_slot in 0..3 {
                for (i, c) in [3, 1, 4, 0, 6, 5, 2].iter().enumerate() {
                    // the actual costas freq at time i
                    let c_fq = (base_fq as f32 + (*c as f32 * 6.25)) as u32;
                    let bin: usize = (c_fq / binwidth) as usize;
                    let energy = (spectra[base_time + (sync_slot * 36 * 4) + (i * 4)][bin].norm());
                    sum = sum + energy;
                }

                // normalizing sum
                // the spectral content of the 7 lowest frequency bins
                for (i, c) in [3, 1, 4, 0, 6, 5, 2].iter().enumerate() {
                    for bin in 0..6 {
                        let energy =
                            (spectra[base_time + (sync_slot * 36 * 4) + (i * 4)][bin].norm());
                        normal = normal + energy;
                    }
                }
            }

            result[[step as usize, base_time as usize]] = sum / normal;
        }
    }

    // result is base_freq x base_time
    // for every freqency, there's a starting time which had the most power
    for (freq_i, f_row) in result.genrows().into_iter().enumerate() {
        let top_time =
            f_row
                .iter()
                .enumerate()
                .fold((freq_i, 0, 0.0), |(freq_i, max_i, max), (i, v)| {
                    if (v > &max) {
                        (freq_i, i, *v)
                    } else {
                        (freq_i, max_i, max)
                    }
                });
        res.push(top_time);
    }

    // hmm maybe not quite, read the pdf...
    let mean: f32 = res
        .iter()
        .map(|(_, _, e)| *e as f32)
        .fold(0.0, |acc, e| acc + e)
        / res.len() as f32;

    println!("mean {}", mean);

    let threshold = 1.50;
    let candidates = res
        .iter()
        .filter(|(_, _, e)| (*e / mean) > threshold)
        .collect::<Vec<_>>();

    println!("length of res {}", res.len());
    println!("length of candidates {}", candidates.len());
    println!("binwidth {}", binwidth);
    println!("candidate (freq_index, time_index, energy) tuples:");
    for c in candidates {
        println!("{:?}", c);
    }

    println!("done");
}

fn find_spectral_peak(filename: &str) -> Option<f32> {
    let mut reader = hound::WavReader::open(filename).expect("Failed to open WAV file");
    let num_samples = reader.len() as usize;
    let mut planner = FFTplanner::new(false);

    let sample_rate = reader.spec().sample_rate;

    let fft = planner.plan_fft(num_samples);

    let mut signal = reader
        .samples::<i16>()
        .map(|x| Complex::new(x.unwrap() as f32, 0f32))
        .collect::<Vec<_>>();
    let mut spectrum = signal.clone();

    fft.process(&mut signal[..], &mut spectrum[..]);

    let max_peak = spectrum
        .iter()
        .take(num_samples / 2)
        .enumerate()
        .max_by_key(|&(_, freq)| freq.norm() as u32);

    if let Some((i, _)) = max_peak {
        let bin = sample_rate as f32 / num_samples as f32;
        Some(i as f32 * bin)
    } else {
        None
    }
}

fn main() {
    let mut reader = hound::WavReader::open("samples/FT8/181201_180245.wav").unwrap();
    let spec = reader.spec();

    println!("Read rate {}", spec.sample_rate);

    // let sqr_sum = reader.samples::<i16>().fold(0.0, |sqr_sum, s| {
    //     let sample = s.unwrap() as f64;
    //     sqr_sum + sample * sample
    // });

    let peak = find_spectral_peak("samples/FT8/191111_110115.wav").unwrap();
    println!("Peak is {} Hz", peak);

    // choppy("samples/FT8/181201_180245.wav");
    choppy("samples/FT8/191111_110115.wav");

    // choppy("sine.wav");
}
