extern crate num;
extern crate rustfft;

use hound;
// use std::f32::consts::PI;
use std::cmp;
use std::i16;

// use rustfft::FFT;
use itertools::Itertools;

use ndarray::Array2;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::FFTplanner;

// png output
use png;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

fn write_wav(bytes: &Vec<u16>, t: usize) -> () {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 12000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(format!("out-{}.wav", t), spec).unwrap();

    for b in bytes {
        writer.write_sample(*b as i16).unwrap();
    }
    // for t in (0..44100).map(|x| x as f32 / 44100.0) {
    //     let sample = (t * 440.0 * 2.0 * PI).sin();
    //     let amplitude = i16::MAX as f32;
    //     writer.write_sample((sample * amplitude) as i16).unwrap();
    // }
    writer.finalize().unwrap();
}

fn spectra_to_bitmap(spectra: &[Vec<Complex<f32>>]) -> Vec<Vec<(u8, u8, u8, u8)>> {
    let height = spectra.len() as u32;
    let width = (spectra[0].len() / 2) as u32;
    let mut max: f32 = 0.0;

    for y in 0..height {
        for x in 0..width {
            if spectra[y as usize][x as usize].norm() > max {
                max = spectra[y as usize][x as usize].norm();
            }
        }
    }

    let mut cols: Vec<Vec<(u8, u8, u8, u8)>> = Vec::new();

    for y in 0..height {
        let mut row: Vec<(u8, u8, u8, u8)> = Vec::new();

        for x in 0..width {
            // normalize to 0,1.0
            let v = spectra[y as usize][x as usize].norm() / max;

            // we'll use an black-green spectrum
            let b = (0.0 * 255.0) as u8;
            let r = (0.0 * 255.0) as u8;
            let g = (v * 255.0) as u8;

            row.push((r, g, b, 255u8));
        }
        cols.push(row);
    }

    cols
}

fn add_candidates_to_bitmap(
    bm: &mut Vec<Vec<(u8, u8, u8, u8)>>,
    candidates: &Vec<(usize, usize, f32)>,
) -> () {
    let mut max: f32 = 0.0;

    for c in candidates {
        match c {
            (_, _, energy) => {
                if *energy > max {
                    max = *energy;
                }
            }
        }
    }

    for c in candidates {
        match c {
            (bin, time, energy) => {
                let a = (255.0 * (*energy as f32 / max as f32)) as u8;
                bm[*time][*bin] = (255, 0, 0, a);

                for sync_slot in 0..3 {
                    for (i, c) in [3, 1, 4, 0, 6, 5, 2].iter().enumerate() {
                        bm[(sync_slot * 36 * 4) + *time + (i * 4)]
                            [*bin + (*c as f32 * 2.0) as usize] = (0, 255, 0, a)
                    }
                }

                bm[(a as f32 / 8.0) as usize][*bin] = (255, 255, 255, 255);
            }
        }
    }

    // add some frequencies to this bitmap too, why not
    let w = bm[0].len();

    // each step is 3.12Hz
    // if we want to make a mark every 100Hz, that's every 100 / 3.125 bins
    let hz_in_bins = 100 as usize;
    let dw = w / (hz_in_bins as f32 / 3.125) as usize;
    for x in 0..dw {
        let bonus = (20 - (x % 10)) * 4;

        for y in 0..(3 + bonus) {
            bm[y][x * (hz_in_bins as f32 / 3.125) as usize] = (255, 255, 0, 255);
        }
    }

    ()
}

fn write_bitmap(filename: &str, input: &Vec<Vec<(u8, u8, u8, u8)>>) {
    let height = input.len() as u32;
    let width = input[0].len() as u32;
    let path = Path::new(filename);
    let file = File::create(path).unwrap();

    let ref mut w = BufWriter::new(file);

    let mut encoder = png::Encoder::new(w, width, height);
    encoder.set_color(png::ColorType::RGBA);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();

    let mut data: Vec<u8> = Vec::new();

    for y in 0..height {
        for x in 0..width {
            match input[y as usize][x as usize] {
                (r, g, b, a) => {
                    data.push(r);
                    data.push(g);
                    data.push(b);
                    data.push(a);
                }
            }
        }
    }

    writer.write_image_data(&data).unwrap(); // Save
}

fn build_spectra(
    signal: &Vec<Complex<f32>>, // original signal
    stepcount: u32,             // number of time steps
    samples_per_step: u32,      // size of sample per time step
    stepsize: u32,              // number of samples to advance every step
) -> Vec<Vec<Complex<f32>>> {
    // make a 2D array of (time-overlapping) FFTs
    println!(
        "time steps {}, time step size {} ({}s)",
        stepcount,
        samples_per_step,
        samples_per_step as f32 / 12000.0
    );

    let mut planner = FFTplanner::new(false);

    // *2 because we're going to pad
    let fft = planner.plan_fft((samples_per_step * 2) as usize);

    let spectra = (0..stepcount)
        .map(|c| {
            let offset = c * stepsize;

            // the input signal to our FFT.
            let mut sig: Vec<Complex<f32>> = vec![Complex::zero(); samples_per_step as usize];

            // the output of our FFT. *2 because we're padding
            let mut spectrum: Vec<Complex<f32>> =
                vec![Complex::zero(); (samples_per_step * 2) as usize];

            // copy a slice of source signal into sig
            sig.copy_from_slice(&signal[offset as usize..(offset + samples_per_step) as usize]);

            // pad the slice with zeros
            let zeros: Vec<Complex<f32>> = vec![Complex::zero(); samples_per_step as usize];
            sig.extend_from_slice(&zeros);

            fft.process(&mut sig, &mut spectrum);

            spectrum
        })
        .collect::<Vec<_>>();

    spectra
}

// returns a vec of tuples of (frequeny_index, max_time_index, power_at_max)
fn find_strongest_costas_for_time(result: &Array2<f32>) -> Vec<(usize, usize, f32)> {
    let mut res = Vec::new();

    for (freq_i, f_row) in result.genrows().into_iter().enumerate() {
        let top_time =
            f_row
                .iter()
                .enumerate()
                .fold((freq_i, 0, 0.0), |(freq_i, max_i, max), (i, v)| {
                    if v > &max {
                        (freq_i, i, *v)
                    } else {
                        (freq_i, max_i, max)
                    }
                });
        res.push(top_time);
    }

    res
}

// for every potential starting point (in time and freq),
// find the "costas value" for that point.
fn find_costas_powers(
    spectra: &Vec<Vec<Complex<f32>>>,
    top_freq: f32,
    binwidth: f32,
) -> Array2<f32> {
    // number of base frequencies we're going to consider
    let steps = (top_freq / binwidth) as u32;

    println!(
        "Frequency steps {} (0 to {}, {}Hz each)",
        steps, top_freq, binwidth
    );

    // spec says 2 seconds before and 3 seconds after, but we don't have that much data!
    // we have exactly 15 seconds of data. message is 12.94 seconds. so we only have an extra 2.06 seconds...
    let start_times = ((1.0 / 0.04) + (1.0 / 0.04)) as usize;
    println!("Start times (time steps): {}", start_times);

    let mut result = Array2::zeros((steps as usize, start_times));

    // it seems like nested `for...` loops are more idiomatic rust, but...
    (0..steps)
        .cartesian_product(0..start_times)
        .for_each(|(fq_step, time_step)| {
            let base_fq = fq_step as f32 * 3.125;
            let base_time = time_step;

            let mut sum = 0.0;
            let mut normal = 0.0;

            for sync_slot in 0..3 {
                for (i, c) in [3, 1, 4, 0, 6, 5, 2].iter().enumerate() {
                    // the costas freq at time i in bin bin
                    let c_fq = base_fq as f32 + (*c as f32 * 6.25);

                    let bin = (c_fq / binwidth) as usize;
                    let energy = spectra[base_time + (sync_slot * 36 * 4) + (i * 4)][bin].norm();

                    sum = sum + energy;
                }

                // normalizing sum
                // the spectral content of the 7 lowest frequency bins
                for (i, _) in [3, 1, 4, 0, 6, 5, 2].iter().enumerate() {
                    for bin in 0..6 {
                        let energy =
                            spectra[base_time + (sync_slot * 36 * 4) + (i * 4)][bin].norm();
                        normal = normal + energy;
                    }
                }
            }

            result[[fq_step as usize, time_step as usize]] = sum / normal;
        });

    result
}

fn costas(
    signal: &Vec<num::Complex<f32>>,
    sample_len: u32,
    sample_rate: u32,
) -> Vec<(usize, usize, f32)> {
    println!("Read rate {}", sample_rate);
    println!(
        "Samples {} so time {}s",
        sample_len,
        sample_len / sample_rate
    );
    let chopsize = (sample_rate as f32 * 0.16) as u32;
    let stepsize = (sample_rate as f32 * 0.04) as u32;

    let chopcount = (sample_len / stepsize) - 3;

    // this is the width in Hz of each element in the FFT result
    let binwidth = sample_rate as f32 / (chopsize * 2) as f32; // * 2 bc we pad

    println!(
        "Covering {}Hz over {} bins, so {} Hz per bin",
        sample_rate / 2,
        chopsize * 2,
        binwidth
    );

    println!("building spectra...");

    // spectra will be a 2D array of (time-overlapping) FFTs
    let spectra = build_spectra(&signal, chopcount, chopsize, stepsize);

    println!("A spectrum is of length {}", spectra[0].len());

    // spectra is a 2D array of [time][freq] -> power
    let mut bm = spectra_to_bitmap(&spectra);

    // these will be our starting frequencies for searching.
    // since each there are 6 costas freq and each is 6.25Hz above the previous,
    // we want to end on (max freq in our passband) - (6 * 6.25)
    let top_of_passband = 5000.0;
    let last_step = top_of_passband; //  - (6.0 * 6.25);

    println!("searching...");

    let result = find_costas_powers(&spectra, last_step, binwidth);

    // result is base_freq x base_time
    // for every frequency, there's a starting time which had the most power.
    // res will be a vec of tuples of
    // (frequency_index, max_time_index, power_at_max)
    let res = find_strongest_costas_for_time(&result);

    // PDF suggests we filter to values 1.5 times over the median.
    // but mean makes more sense to me? anywhere here's both.
    let mean: f32 = res
        .iter()
        .map(|(_, _, e)| *e as f32)
        .fold(0.0, |acc, e| acc + e)
        / res.len() as f32;

    let ordered = res
        .iter()
        .sorted_by(|(_, _, a), (_, _, b)| b.partial_cmp(a).unwrap())
        .collect::<Vec<_>>();

    let median = ordered[(res.len() / 2) as usize];

    //let threshold = median.2 * 1.5;
    let threshold = mean * 1.5;
    println!("threshold {}", threshold);

    // let candidates = res
    //     .iter()
    //     .filter(|(_, _, e)| (*e / mean) > threshold)
    //     .copied()
    //     .collect::<Vec<_>>();

    // take local maxima of those!
    let mut maxima = Vec::new();
    let mut last_power: Option<f32> = None;
    let mut did_push: bool = false;

    for i in 0..res.len() - 1 {
        match i {
            0 => {
                if res[0].2 > res[1].2 && (res[i].2 / mean) > threshold {
                    maxima.push(res[0].clone());
                }
            }
            _ => {
                if res[i - 1].2 < res[i].2
                    && res[i].2 > res[i + 1].2
                    && (res[i].2 / mean) > threshold
                {
                    did_push = true;
                    maxima.push(res[i].clone());
                } else if did_push && res[i].2 == res[i + 1].2 && (res[i].2 / mean) > threshold {
                    did_push = true;
                    maxima.push(res[i].clone());
                } else {
                    did_push = false;
                }
            }
        }
    }
    println!("maxima {:?}", maxima);

    let as_freq = maxima
        .iter()
        .map(|(f, t, e)| (*f as f32 * binwidth, *e))
        .collect::<Vec<_>>();

    // add_candidates_to_bitmap(&mut bm, &as_freq);
    // add_candidates_to_bitmap(&mut bm, &candidates);
    add_candidates_to_bitmap(&mut bm, &maxima);

    // println!("length of res {}", res.len());
    // println!("length of candidates {}", maxima.len());

    println!("done");

    write_bitmap("waterfall.png", &bm);

    maxima
}

// fn fine_process(reader: &mut hound::WavReader<std::io::BufReader<std::fs::File>>) -> () {
fn fine_process(
    original_signal: &Vec<num::Complex<f32>>,
    num_samples: u32,
    sample_rate: u32,
    candidates: &Vec<(usize, usize, f32)>, // (fq bin, time, power) triples
) -> () {
    let mut signal = original_signal.clone();

    // fft of entire signal
    let mut planner = FFTplanner::new(false);
    let mut iplanner = FFTplanner::new(true);

    // one second of zeros
    let zeros: Vec<Complex<f32>> = vec![Complex::zero(); sample_rate as usize];

    signal.extend_from_slice(&zeros);
    let mut spectrum = signal.clone();

    let total_samples = spectrum.len();
    let bins = total_samples;

    let fft = planner.plan_fft(spectrum.len() as usize);
    let ifft = iplanner.plan_fft(spectrum.len() as usize);

    fft.process(&mut signal[..], &mut spectrum[..]);

    // ok, for each candidate base frequency f,
    // we chop out f-6.25 to f+50 from the full fft
    // let binwidth = 3.125;
    let binwidth = (sample_rate as f32 / 2.0) / total_samples as f32;
    println!(
        "BINWIDTH {} ({} bins) spectrum length {}",
        binwidth,
        bins,
        spectrum.len()
    );
    for (f, t, p) in candidates {
        let min = cmp::max(
            0,
            (*f as f32 / binwidth) as usize - ((6.25 / binwidth) as usize),
        );
        let max = (*f as f32 / binwidth) as usize + ((50.0 / binwidth) as usize); // XXX jt overflow
                                                                                  // xxx jt you are here
        let mut spectrum_slice_0: Vec<Complex<f32>> = vec![Complex::zero(); min];
        let mut spectrum_slice_1: Vec<Complex<f32>> = vec![Complex::zero(); max - min];
        let mut spectrum_slice_2: Vec<Complex<f32>> =
            vec![Complex::zero(); signal.len() as usize - max];

        let mut generated_signal = vec![Complex::zero(); signal.len()];

        spectrum_slice_1.copy_from_slice(&spectrum[min..max]);
        spectrum_slice_0.extend(spectrum_slice_1);
        spectrum_slice_0.extend(spectrum_slice_2);

        let mut spectrum_copy: Vec<Complex<f32>> = spectrum.clone();

        ifft.process(&mut spectrum_slice_0, &mut generated_signal);
        // ifft.process(&mut spectrum_copy, &mut generated_signal);

        let reals = generated_signal[..generated_signal.len() as usize]
            .into_iter()
            .map(|v| ((v.re / (generated_signal.len() as f32)) as u16))
            .collect::<Vec<u16>>();

        write_wav(&reals, *f);
    }
}

fn choppy(fname: &str) -> (Vec<num::Complex<f32>>, u32, u32) {
    let mut reader = hound::WavReader::open(fname).expect("Failed to open WAV file");

    let signal = reader
        .samples::<i16>()
        .map(|x| Complex::new(x.unwrap() as f32, 0f32))
        .collect::<Vec<_>>();

    (signal, reader.len(), reader.spec().sample_rate)
}

fn main() {
    // let mut reader = hound::WavReader::open("samples/FT8/181201_180245.wav").unwrap();
    // let spec = reader.spec();

    // let sqr_sum = reader.samples::<i16>().fold(0.0, |sqr_sum, s| {
    //     let sample = s.unwrap() as f64;
    //     sqr_sum + sample * sample
    // });

    let (signal, sample_len, rate) =
    // choppy("samples/FT8/181201_180245.wav");
    // choppy("../ft8_lib/tests/191111_110130.wav");
    // choppy("../ft8_lib/tests/191111_110145.wav");
    //  choppy("../ft8_lib/tests/191111_110630.wav");
    // choppy("../ft8_lib/tests/191111_110645.wav");
    // choppy("../ft8_lib/tests/191111_110200.wav");
    // choppy("../ft8_lib/tests/191111_110215.wav");
    choppy("../ft8_lib/tests/191111_110615.wav");
    // choppy("3-4-5000hz.wav");
    // choppy("samples/FT8/191111_110115.wav");

    let mut candidates: std::vec::Vec<(usize, usize, f32)> = costas(&signal, sample_len, rate);
    candidates.sort_by(|(_, _, l), (_, _, r)| r.partial_cmp(l).unwrap());
    // println!("{:?}", candidates);

    fine_process(&signal, sample_len, rate, &candidates);

    // choppy("sine.wav");
}
