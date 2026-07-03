use std::{
    error::Error,
    fs,
    fs::File,
    io::Write,
    path::{Path, PathBuf},
};

use flate2::read::GzDecoder;

const BASE_URL: &str = "https://storage.googleapis.com/cvdf-datasets/mnist";
const IMAGE_ROWS: usize = 28;
const IMAGE_COLS: usize = 28;
const IMAGE_PIXELS: usize = IMAGE_ROWS * IMAGE_COLS;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub struct Mnist {
    images: Vec<f32>,
    labels: Vec<u8>,
}

impl Mnist {
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn image(&self, index: usize) -> &[f32] {
        let start = index * IMAGE_PIXELS;
        &self.images[start..start + IMAGE_PIXELS]
    }

    pub fn label(&self, index: usize) -> u8 {
        self.labels[index]
    }
}

pub fn load_or_download(dir: impl AsRef<Path>) -> Result<(Mnist, Mnist)> {
    let dir = dir.as_ref();
    fs::create_dir_all(dir)?;

    let train_images = ensure_file(dir, "train-images-idx3-ubyte")?;
    let train_labels = ensure_file(dir, "train-labels-idx1-ubyte")?;
    let test_images = ensure_file(dir, "t10k-images-idx3-ubyte")?;
    let test_labels = ensure_file(dir, "t10k-labels-idx1-ubyte")?;

    Ok((
        Mnist {
            images: read_images(&train_images)?,
            labels: read_labels(&train_labels)?,
        },
        Mnist {
            images: read_images(&test_images)?,
            labels: read_labels(&test_labels)?,
        },
    ))
}

fn ensure_file(dir: &Path, name: &str) -> Result<PathBuf> {
    let raw_path = dir.join(name);
    if raw_path.exists() {
        return Ok(raw_path);
    }

    let gz_name = format!("{name}.gz");
    let gz_path = dir.join(&gz_name);
    if !gz_path.exists() {
        let url = format!("{BASE_URL}/{gz_name}");
        eprintln!("downloading {url}");
        let response = ureq::get(&url).call()?;
        let mut reader = response.into_reader();
        let mut file = File::create(&gz_path)?;
        std::io::copy(&mut reader, &mut file)?;
        file.flush()?;
    }

    let mut decoder = GzDecoder::new(File::open(&gz_path)?);
    let mut file = File::create(&raw_path)?;
    std::io::copy(&mut decoder, &mut file)?;
    Ok(raw_path)
}

fn read_images(path: &Path) -> Result<Vec<f32>> {
    let bytes = fs::read(path)?;
    if read_u32(&bytes, 0)? != 2051 {
        return Err(format!("{} is not an IDX image file", path.display()).into());
    }
    let count = read_u32(&bytes, 4)? as usize;
    let rows = read_u32(&bytes, 8)? as usize;
    let cols = read_u32(&bytes, 12)? as usize;
    if rows != IMAGE_ROWS || cols != IMAGE_COLS {
        return Err(format!("expected 28x28 images, got {rows}x{cols}").into());
    }

    let expected = 16 + count * rows * cols;
    if bytes.len() != expected {
        return Err(format!(
            "{} has {} bytes, expected {expected}",
            path.display(),
            bytes.len()
        )
        .into());
    }

    Ok(bytes[16..]
        .iter()
        .map(|pixel| f32::from(*pixel) / 255.0)
        .collect())
}

fn read_labels(path: &Path) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    if read_u32(&bytes, 0)? != 2049 {
        return Err(format!("{} is not an IDX label file", path.display()).into());
    }
    let count = read_u32(&bytes, 4)? as usize;
    let expected = 8 + count;
    if bytes.len() != expected {
        return Err(format!(
            "{} has {} bytes, expected {expected}",
            path.display(),
            bytes.len()
        )
        .into());
    }
    Ok(bytes[8..].to_vec())
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let end = offset + 4;
    let value = bytes
        .get(offset..end)
        .ok_or_else(|| format!("truncated IDX header at byte {offset}"))?;
    Ok(u32::from_be_bytes(value.try_into()?))
}
