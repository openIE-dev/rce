//! TLS between hosts for P4 channels (W1i/P4): TLS 1.3 only, mutual, with
//! no certificate authority. Each host has one Ed25519 identity; its
//! certificate is self-signed (rcgen) on that key, and a peer is accepted
//! only when the key in its certificate equals the key pinned for it
//! (`--peer HOST:PORT=ED25519HEX`). The host logs its public key in D as
//! `host.TlsIdentity` at start, so the identity a run used is part of the
//! record; the 32-byte seed stays in `<log>.tlskey` (mode 0600), never in D.
//! Pinning is a byte comparison of keys; nothing here asks a model.
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::{CryptoProvider, verify_tls13_signature};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName, UnixTime};
use rustls::server::danger::{ClientCertVerified, ClientCertVerifier};
use rustls::{ClientConfig, DigitallySignedStruct, DistinguishedName, Error, ServerConfig, SignatureScheme};
use std::path::Path;
use std::sync::Arc;

pub const SERVER_NAME: &str = "rce-peer";

#[derive(Clone)]
pub struct Identity {
    pub seed: [u8; 32],
    pub public: [u8; 32],
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

pub fn unhex32(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 || !s.bytes().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let mut o = [0u8; 32];
    for i in 0..32 {
        o[i] = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(o)
}

fn pkcs8(seed: &[u8; 32]) -> Vec<u8> {
    // RFC 8410 OneAsymmetricKey for Ed25519: fixed prefix, then the seed
    let mut v = vec![0x30, 0x2e, 0x02, 0x01, 0x00, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x04, 0x22, 0x04, 0x20];
    v.extend_from_slice(seed);
    v
}

impl Identity {
    pub fn from_seed(seed: [u8; 32]) -> Result<Identity, String> {
        use ring::signature::KeyPair;
        let kp = ring::signature::Ed25519KeyPair::from_seed_unchecked(&seed).map_err(|e| e.to_string())?;
        let mut public = [0u8; 32];
        public.copy_from_slice(kp.public_key().as_ref());
        Ok(Identity { seed, public })
    }

    /// The identity kept beside the log `D` in `D.tlskey`, made from the OS
    /// CSPRNG the first time.
    pub fn load_or_create(log: &Path) -> Result<Identity, String> {
        let p = log.with_extension(match log.extension() {
            Some(e) => format!("{}.tlskey", e.to_string_lossy()),
            None => "tlskey".into(),
        });
        let seed: [u8; 32] = match std::fs::read(&p) {
            Ok(b) if b.len() == 32 => b.try_into().unwrap(),
            Ok(_) => return Err(format!("{}: not a 32-byte key", p.display())),
            Err(_) => {
                let mut s = [0u8; 32];
                std::io::Read::read_exact(&mut std::fs::File::open("/dev/urandom").map_err(|e| e.to_string())?, &mut s).map_err(|e| e.to_string())?;
                let mut o = std::fs::OpenOptions::new();
                o.write(true).create_new(true);
                #[cfg(unix)]
                std::os::unix::fs::OpenOptionsExt::mode(&mut o, 0o600);
                std::io::Write::write_all(&mut o.open(&p).map_err(|e| format!("{}: {e}", p.display()))?, &s).map_err(|e| e.to_string())?;
                s
            }
        };
        Identity::from_seed(seed)
    }

    fn cert(&self) -> Result<(CertificateDer<'static>, PrivateKeyDer<'static>), String> {
        let der = PrivatePkcs8KeyDer::from(pkcs8(&self.seed));
        let kp = rcgen::KeyPair::from_pkcs8_der_and_sign_algo(&der, &rcgen::PKCS_ED25519).map_err(|e| e.to_string())?;
        let params = rcgen::CertificateParams::new(vec![SERVER_NAME.to_string()]).map_err(|e| e.to_string())?;
        let c = params.self_signed(&kp).map_err(|e| e.to_string())?;
        Ok((c.der().clone(), PrivateKeyDer::Pkcs8(der)))
    }
}

/// The Ed25519 key in a certificate's SubjectPublicKeyInfo:
/// SEQUENCE { SEQUENCE { OID 1.3.101.112 } BIT STRING (0 unused) 32 bytes }.
pub fn ed25519_key_of(cert: &[u8]) -> Option<[u8; 32]> {
    let pat = [0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x03, 0x21, 0x00];
    let i = cert.windows(pat.len()).position(|w| w == pat)?;
    cert.get(i + pat.len()..i + pat.len() + 32)?.try_into().ok()
}

fn provider() -> Arc<CryptoProvider> {
    Arc::new(rustls::crypto::ring::default_provider())
}

#[derive(Debug)]
struct Pins {
    keys: Vec<[u8; 32]>,
    prov: Arc<CryptoProvider>,
}

impl Pins {
    fn check(&self, c: &CertificateDer<'_>) -> Result<(), Error> {
        match ed25519_key_of(c) {
            Some(k) if self.keys.contains(&k) => Ok(()),
            _ => Err(Error::General("peer key is not pinned".into())),
        }
    }
    fn sig(&self, m: &[u8], c: &CertificateDer<'_>, d: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, Error> {
        if d.scheme != SignatureScheme::ED25519 {
            return Err(Error::General("only Ed25519".into()));
        }
        verify_tls13_signature(m, c, d, &self.prov.signature_verification_algorithms)
    }
}

impl ServerCertVerifier for Pins {
    fn verify_server_cert(&self, e: &CertificateDer<'_>, _: &[CertificateDer<'_>], _: &ServerName<'_>, _: &[u8], _: UnixTime) -> Result<ServerCertVerified, Error> {
        self.check(e).map(|_| ServerCertVerified::assertion())
    }
    fn verify_tls12_signature(&self, _: &[u8], _: &CertificateDer<'_>, _: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, Error> {
        Err(Error::General("TLS 1.2 is not used".into()))
    }
    fn verify_tls13_signature(&self, m: &[u8], c: &CertificateDer<'_>, d: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, Error> {
        self.sig(m, c, d)
    }
    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        vec![SignatureScheme::ED25519]
    }
}

impl ClientCertVerifier for Pins {
    fn root_hint_subjects(&self) -> &[DistinguishedName] {
        &[]
    }
    fn verify_client_cert(&self, e: &CertificateDer<'_>, _: &[CertificateDer<'_>], _: UnixTime) -> Result<ClientCertVerified, Error> {
        self.check(e).map(|_| ClientCertVerified::assertion())
    }
    fn verify_tls12_signature(&self, _: &[u8], _: &CertificateDer<'_>, _: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, Error> {
        Err(Error::General("TLS 1.2 is not used".into()))
    }
    fn verify_tls13_signature(&self, m: &[u8], c: &CertificateDer<'_>, d: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, Error> {
        self.sig(m, c, d)
    }
    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        vec![SignatureScheme::ED25519]
    }
}

/// Accepts clients whose key is one of `pins`.
pub fn server_config(id: &Identity, pins: Vec<[u8; 32]>) -> Result<Arc<ServerConfig>, String> {
    let prov = provider();
    let (c, k) = id.cert()?;
    let v = Arc::new(Pins { keys: pins, prov: prov.clone() });
    let cfg = ServerConfig::builder_with_provider(prov)
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(|e| e.to_string())?
        .with_client_cert_verifier(v)
        .with_single_cert(vec![c], k)
        .map_err(|e| e.to_string())?;
    Ok(Arc::new(cfg))
}

/// Connects only to a server whose key is `pin`.
pub fn client_config(id: &Identity, pin: [u8; 32]) -> Result<Arc<ClientConfig>, String> {
    let prov = provider();
    let (c, k) = id.cert()?;
    let v = Arc::new(Pins { keys: vec![pin], prov: prov.clone() });
    let cfg = ClientConfig::builder_with_provider(prov)
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(|e| e.to_string())?
        .dangerous()
        .with_custom_certificate_verifier(v)
        .with_client_auth_cert(vec![c], k)
        .map_err(|e| e.to_string())?;
    Ok(Arc::new(cfg))
}
