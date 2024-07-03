use num::Complex;
use std::str::FromStr;
use image::ColorType;
use image::png::PNGEncoder;
use std::fs::File;
use std::env;

static LIMIT: usize = 100;

static VGA_PALETTE_LENGTH: i32 = 256;

static VGA_PALETTE: [&str; 256] = [
    "000000", "0002aa", "14aa00", "00aaaa", "aa0003", "aa00aa", "aa5500", "aaaaaa",
    "555555", "5555ff", "55ff55", "55ffff", "ff5555", "fd55ff", "ffff55", "ffffff",
    "000000", "101010", "202020", "353535", "454545", "555555", "656565", "757575",
    "8a8a8a", "9a9a9a", "aaaaaa", "bababa", "cacaca", "dfdfdf", "efefef", "ffffff",
    "0004ff", "4104ff", "8203ff", "be02ff", "fd00ff", "fe00be", "ff0082", "ff0041",
    "ff0008", "ff4105", "ff8200", "ffbe00", "ffff00", "beff00", "82ff00", "41ff01",
    "24ff00", "22ff42", "1dff82", "12ffbe", "00ffff", "00beff", "0182ff", "0041ff",
    "8282ff", "9e82ff", "be82ff", "df82ff", "fd82ff", "fe82df", "ff82be", "ff829e",
    "ff8282", "ff9e82", "ffbe82", "ffdf82", "ffff82", "dfff82", "beff82", "9eff82",
    "82ff82", "82ff9e", "82ffbe", "82ffdf", "82ffff", "82dfff", "82beff", "829eff",
    "babaff", "cabaff", "dfbaff", "efbaff", "febaff", "febaef", "ffbadf", "ffbaca",
    "ffbaba", "ffcaba", "ffdfba", "ffefba", "ffffba", "efffba", "dfffba", "caffbb",
    "baffba", "baffca", "baffdf", "baffef", "baffff", "baefff", "badfff", "bacaff",
    "010171", "1c0171", "390171", "550071", "710071", "710055", "710039", "71001c",
    "710001", "711c01", "713900", "715500", "717100", "557100", "397100", "1c7100",
    "097100", "09711c", "067139", "037155", "007171", "005571", "003971", "001c71",
    "393971", "453971", "553971", "613971", "713971", "713961", "713955", "713945",
    "713939", "714539", "715539", "716139", "717139", "617139", "557139", "45713a",
    "397139", "397145", "397155", "397161", "397171", "396171", "395571", "394572",
    "515171", "595171", "615171", "695171", "715171", "715169", "715161", "715159",
    "715151", "715951", "716151", "716951", "717151", "697151", "617151", "597151",
    "517151", "51715a", "517161", "517169", "517171", "516971", "516171", "515971",
    "000042", "110041", "200041", "310041", "410041", "410032", "410020", "410010",
    "410000", "411000", "412000", "413100", "414100", "314100", "204100", "104100",
    "034100", "034110", "024120", "014131", "004141", "003141", "002041", "001041",
    "202041", "282041", "312041", "392041", "412041", "412039", "412031", "412028",
    "412020", "412820", "413120", "413921", "414120", "394120", "314120", "284120",
    "204120", "204128", "204131", "204139", "204141", "203941", "203141", "202841",
    "2d2d41", "312d41", "352d41", "3d2d41", "412d41", "412d3d", "412d35", "412d31",
    "412d2d", "41312d", "41352d", "413d2d", "41412d", "3d412d", "35412d", "31412d",
    "2d412d", "2d4131", "2d4135", "2d413d", "2d4141", "2d3d41", "2d3541", "2d3141",
    "000000", "000000", "000000", "000000", "000000", "000000", "000000", "000000",
];

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 5 {
        eprintln!("Usage: {} FILE PIXELS UPPERLEFT LOWERRIGHT",
                  args[0]);
        eprintln!("Example: {} mandel.png 1000x750 -1.20,0.35 -1,0.20",
                  args[0]);
        std::process::exit(1);
    }

    let bounds = parse_pair(&args[2], 'x')
        .expect("error parsing image dimensions");
    let upper_left = parse_complex(&args[3])
        .expect("error parsing upper left corner point");
    let lower_right = parse_complex(&args[4])
        .expect("error parsing lower right corner point");

    let mut pixels = vec![0; bounds.0 * bounds.1];

    // Single threaded
    // render(&mut pixels, bounds, upper_left, lower_right);

    let threads = 8;
    // Round row count upward to make sure bands cover the entire image even if height isn't multiple of threads
    let rows_per_band = bounds.1 / threads + 1;

    {
        // Divide the pixel buffer into bands
        let bands: Vec<&mut [u8]> = pixels.chunks_mut(rows_per_band * bounds.0).collect();

        // Start a new scope that'll wait for all the threads within to finish
        crossbeam::scope(|spawner| {
           for (i, band) in bands.into_iter().enumerate() {
               // Produce bounding boxes needed to call the renderer for the section
               let top = rows_per_band * i;
               let height = band.len() / bounds.0;
               let band_bounds = (bounds.0, height);
               let band_upper_left = pixel_to_point(bounds, (0, top), upper_left, lower_right);
               let band_lower_right = pixel_to_point(bounds, (bounds.0, top + height), upper_left, lower_right);
               // Create a new thread to render the section
               spawner.spawn(move |_| {
                   render(band, band_bounds, band_upper_left, band_lower_right);
               });
           }
        }).unwrap();
    }

    write_image(&args[1], &pixels, bounds)
        .expect("error writing PNG file");
}

/// Try to determine if `c` is in the Mandelbrot set, using at most `limit`
/// iterations to decide.
///
/// If `c` is not a member, return `Some(i)`, where `i` is the number of
/// iterations it took for `c` to leave the circle of radius two centered on the
/// origin. If `c` seems to be a member (more precisely, if we reached the
/// iteration limit without being able to prove that `c` is not a member),
/// return `None`.
fn escape_time(c: Complex<f64>, limit: usize) -> Option<usize> {
    let mut z = Complex { re: 0.0, im: 0.0 };

    for i in 0..limit {
        if z.norm_sqr() > 4.0 {
            return Some(i);
        }
        z = z * z + c;
    }

    None
}

/// Parse the string `s` as a coordinate pair, like `"400x600"` or `"1.0,0.5"`.
///
/// Specifically, `s` should have the form <left><sep><right>, where <sep> is
/// the character given by the `separator` argument, and <left> and <right> are both
/// strings that can be parsed by `T::from_str`.
///
/// If `s` has the proper form, return `Some<(x, y)>`. If it doesn't parse
/// correctly, return `None`.
fn parse_pair<T: FromStr>(s: &str, separator: char) -> Option<(T, T)> {
    match s.find(separator) {
        None => None,
        Some(index) => {
            match (T::from_str(&s[..index]), T::from_str(&s[index + 1..])) {
                (Ok(l), Ok(r)) => Some((l, r)),
                _ => None
            }
        }
    }
}

#[test]
fn test_parse_pair() {
    assert_eq!(parse_pair::<i32>("",        ','), None);
    assert_eq!(parse_pair::<i32>("10,",     ','), None);
    assert_eq!(parse_pair::<i32>(",10",     ','), None);
    assert_eq!(parse_pair::<i32>("10,20",   ','), Some((10, 20)));
    assert_eq!(parse_pair::<i32>("10,20xy", ','), None);
    assert_eq!(parse_pair::<f64>("0.5x",    'x'), None);
    assert_eq!(parse_pair::<f64>("0.5x1.5", 'x'), Some((0.5, 1.5)));
}

/// Parse a pair of floating-point numbers separated by a comma as a complex
/// number.
fn parse_complex(s: &str) -> Option<Complex<f64>> {
    match parse_pair(s, ',') {
        Some((re, im)) => Some(Complex { re, im }),
        None => None
    }
}

#[test]
fn test_parse_complex() {
    assert_eq!(parse_complex("1.25,-0.0625"),
               Some(Complex { re: 1.25, im: -0.0625 }));
    assert_eq!(parse_complex(",-0.0625"), None);
}

/// Given the row and column of a pixel in the output image, return the
/// corresponding point on the complex plane.
///
/// `bounds` is a pair giving the width and height of the image in pixels.
/// `pixel` is a (column, row) pair indicating a particular pixel in that image.
/// The `upper_left` and `lower_right` parameters are points on the complex
/// plane designating the area our image covers.
fn pixel_to_point(bounds: (usize, usize),
                  pixel: (usize, usize),
                  upper_left: Complex<f64>,
                  lower_right: Complex<f64>)
                  -> Complex<f64>
{
    let (width, height) = (lower_right.re - upper_left.re,
                           upper_left.im - lower_right.im);
    Complex {
        re: upper_left.re + pixel.0 as f64 * width  / bounds.0 as f64,
        im: upper_left.im - pixel.1 as f64 * height / bounds.1 as f64
        // Why subtraction here? pixel.1 increases as we go down,
        // but the imaginary component increases as we go up.
    }
}

#[test]
fn test_pixel_to_point() {
    assert_eq!(pixel_to_point((100, 200), (25, 175),
                              Complex { re: -1.0, im:  1.0 },
                              Complex { re:  1.0, im: -1.0 }),
               Complex { re: -0.5, im: -0.75 });
}

/// Render a rectangle of the Mandelbrot set into a buffer of pixels.
///
/// The `bounds` argument gives the width and height of the buffer `pixels`,
/// which holds one grayscale pixel per byte. The `upper_left` and `lower_right`
/// arguments specify points on the complex plane corresponding to the upper-
/// left and lower-right corners of the pixel buffer.
fn render(pixels: &mut [u8],
          bounds: (usize, usize),
          upper_left: Complex<f64>,
          lower_right: Complex<f64>)
{
    assert!(pixels.len() == bounds.0 * bounds.1);

    for row in 0..bounds.1 {
        for column in 0..bounds.0 {
            let point = pixel_to_point(bounds, (column, row),
                                       upper_left, lower_right);
            pixels[row * bounds.0 + column] =
                match escape_time(point, LIMIT) {
                    None => LIMIT,
                    Some(count) => count as u8
                };
        }
    }
}

fn colorize_pixels(pixels: &[u8]) -> Vec<u8> {
    let mut result = vec![0; pixels.len() * 3];

    let mut i: usize = 0;
    for pixel in pixels {
        let (r, g, b) = hex_to_rgb(VGA_PALETTE[32 + *pixel as usize]);

        result[i] = r;
        result[i + 1] = g;
        result[i + 2] = b;

        i += 3;
    }

    result
}

fn hex_to_rgb(hex: &str) -> (u8, u8, u8) {
    (u8::from_str_radix(&hex[..2], 16).unwrap(),
     u8::from_str_radix(&hex[2..4], 16).unwrap(),
     u8::from_str_radix(&hex[4..], 16).unwrap())
}


/// Write the buffer `pixels`, whose dimensions are given by `bounds`, to the
/// file named `filename`.
fn write_image(filename: &str, pixels: &[u8], bounds: (usize, usize))
               -> Result<(), std::io::Error>
{
    let output = File::create(filename)?;

    let pixels = colorize_pixels(pixels);

    let encoder = PNGEncoder::new(output);
    encoder.encode(&pixels,
                   bounds.0 as u32, bounds.1 as u32,
                   ColorType::RGB(8))?;

    Ok(())
}