extern crate num;
extern crate rustfft;

use hound;
// use std::f32::consts::PI;
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

// fn write_wav() -> () {
//     let spec = hound::WavSpec {
//         channels: 1,
//         sample_rate: 44100,
//         bits_per_sample: 16,
//         sample_format: hound::SampleFormat::Int,
//     };
//     let mut writer = hound::WavWriter::create("sine.wav", spec).unwrap();
//     for t in (0..44100).map(|x| x as f32 / 44100.0) {
//         let sample = (t * 440.0 * 2.0 * PI).sin();
//         let amplitude = i16::MAX as f32;
//         writer.write_sample((sample * amplitude) as i16).unwrap();
//     }
//     writer.finalize().unwrap();
// }

fn spectra_to_bitmap(spectra: &[Vec<Complex<f32>>]) -> Vec<Vec<(u8, u8, u8, u8)>> {
    let height = spectra.len() as u32;
    let width = (spectra[0].len() / 2) as u32;

    let mut max: f32 = 0.0;
    for y in 0..height {
        for x in 0..width {
            if spectra[y as usize][x as usize].norm_sqr() > max {
                max = spectra[y as usize][x as usize].norm_sqr();
            }
        }
    }

    let mut cols: Vec<Vec<(u8, u8, u8, u8)>> = Vec::new();

    for y in 0..height {
        let mut row: Vec<(u8, u8, u8, u8)> = Vec::new();
        for x in 0..width {
            let v = ((spectra[y as usize][x as usize].norm_sqr() / max) * 255.0) as u8;
            row.push((v / 2, v / 2, v, 255u8));
        }
        cols.push(row);
    }

    cols
}

fn add_candidates_to_bitmap(
    bm: &mut Vec<Vec<(u8, u8, u8, u8)>>,
    candidates: &Vec<&(usize, usize, f32)>,
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
            (freq, time, energy) => {
                let a = (255.0 * (*energy as f32 / max as f32)) as u8;
                bm[*time][*freq] = (255, 0, 0, a);

                for sync_slot in 0..3 {
                    for (i, c) in [3, 1, 4, 0, 6, 5, 2].iter().enumerate() {
                        bm[(sync_slot * 36 * 4) + *time + (i * 4)]
                            [*freq + (*c as f32 * 2.0) as usize] = (0, 255, 0, a)
                    }
                }
                bm[(a as f32 / 8.0) as usize][*freq] = (0, 0, 255, 255);
            }
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
    signal: &Vec<Complex<f32>>,
    chopcount: u32,
    chopsize: u32,
    stepsize: u32,
) -> Vec<Vec<Complex<f32>>> {
    // make a 2D array of (time-overlapping) FFTs
    println!("chopcount {}", chopcount);
    let mut planner = FFTplanner::new(false);

    let fft = planner.plan_fft((chopsize * 2) as usize);

    let spectra = (0..chopcount)
        .map(|c| {
            let offset = c * stepsize;

            let mut sig: Vec<Complex<f32>> = vec![Complex::zero(); chopsize as usize];
            let mut spectrum: Vec<Complex<f32>> = vec![Complex::zero(); (chopsize * 2) as usize];

            sig.copy_from_slice(&signal[offset as usize..(offset + chopsize) as usize]);
            let zeros: Vec<Complex<f32>> = vec![Complex::zero(); chopsize as usize];

            sig.extend_from_slice(&zeros);

            fft.process(&mut sig, &mut spectrum[..]);

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
    let steps = (top_freq / 3.125) as u32;

    println!("Frequency steps {}", steps);

    // spec says 2 seconds before and 3 seconds after, but we don't have that much data!
    // we have exactly 15 seconds of data. message is 12.94 seconds. so we only have an extra 2.06 seconds...
    let start_times = ((1.0 / 0.04) + (1.0 / 0.04)) as usize;
    println!("Start times (time steps): {}", start_times);

    let mut result = Array2::zeros((steps as usize, start_times));

    // it seems like nested `for...` loops are more idiomatic rust, but...
    (0..steps)
        .cartesian_product(0..start_times)
        .for_each(|(fq_step, time_step)| {
            // for-each is side-effecting/non-lazy map

            let base_fq = fq_step as f32 * 3.125;
            let base_time = time_step;

            let mut sum = 0.0;
            let mut normal = 0.0;

            for sync_slot in 0..3 {
                for (i, c) in [3, 1, 4, 0, 6, 5, 2].iter().enumerate() {
                    // the costas freq at time i
                    let c_fq = base_fq as f32 + (*c as f32 * 6.25);

                    let bin = (c_fq / binwidth) as usize;
                    let energy =
                        spectra[base_time + (sync_slot * 36 * 4) + (i * 4)][bin].norm_sqr();

                    sum = sum + energy;
                }

                // normalizing sum
                // the spectral content of the 7 lowest frequency bins
                for (i, _) in [3, 1, 4, 0, 6, 5, 2].iter().enumerate() {
                    for bin in 0..6 {
                        let energy =
                            spectra[base_time + (sync_slot * 36 * 4) + (i * 4)][bin].norm_sqr();
                        normal = normal + energy;
                    }
                }
            }

            result[[fq_step as usize, time_step as usize]] = sum / normal;
        });

    result
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
    // our nyquist freq is sample_rate / 2, so 6kHz
    // our FFT will have `chopsize` * 2 bins (`chopsize` usable)
    let binwidth = (spec.sample_rate as f32 / 2.0) / chopsize as f32;

    println!(
        "Covering {}Hz over {} bins, so {} Hz per bin",
        spec.sample_rate / 2,
        chopsize,
        binwidth
    );

    let signal = reader
        .samples::<i16>()
        .map(|x| Complex::new(x.unwrap() as f32, 0f32))
        .collect::<Vec<_>>();

    println!("building spectra...");

    // spectra will be a 2D array of (time-overlapping) FFTs
    let spectra = build_spectra(&signal, chopcount, chopsize, stepsize);

    println!(
        "A spectrum is of length {} ({} reals)",
        spectra[0].len(),
        spectra[0].len() / 2
    );

    // spectra is a 2D array of [time][freq] -> power
    let mut bm = spectra_to_bitmap(&spectra);

    // these will be our starting frequencies for searching.
    // since each there are 6 costas freq and each is 6.25Hz above the previous,
    // we want to end on (max freq in our passband) - (6 * 6.25)
    let top_of_passband = 2600.0;
    let last_step = top_of_passband - (6.0 * 6.25);

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

    // let threshold = 10.0;
    let threshold = median.2 * 1.5;
    println!("threshold {}", threshold);
    let candidates = res
        .iter()
        .filter(|(_, _, e)| (*e / mean) > threshold)
        .collect::<Vec<_>>();

    add_candidates_to_bitmap(&mut bm, &candidates);

    println!("length of res {}", res.len());
    println!("length of candidates {}", candidates.len());

    println!("done");

    write_bitmap("waterfall.png", &bm);
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
    // let mut reader = hound::WavReader::open("samples/FT8/181201_180245.wav").unwrap();
    // let spec = reader.spec();

    // let sqr_sum = reader.samples::<i16>().fold(0.0, |sqr_sum, s| {
    //     let sample = s.unwrap() as f64;
    //     sqr_sum + sample * sample
    // });

    let peak = find_spectral_peak("samples/FT8/191111_110115.wav").unwrap();
    println!("Peak is {} Hz", peak);

    // choppy("samples/FT8/181201_180245.wav");
    choppy("../ft8_lib/tests/191111_110130.wav");
    // choppy("../ft8_lib/tests/191111_110145.wav");
    // choppy("../ft8_lib/tests/191111_110630.wav");
    // choppy("../ft8_lib/tests/191111_110645.wav");
    // choppy("../ft8_lib/tests/191111_110200.wav");
    // choppy("../ft8_lib/tests/191111_110215.wav");

    // choppy("samples/FT8/191111_110115.wav");

    // choppy("sine.wav");
}
