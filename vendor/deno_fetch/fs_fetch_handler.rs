// Copyright 2018-2026 the Deno authors. MIT license.

use std::borrow::Cow;
use std::cell::RefCell;
use std::rc::Rc;

use deno_core::futures::FutureExt;
use deno_core::url::Url;
use deno_core::CancelFuture;
use deno_core::OpState;
use deno_fs::AccessCheckFn;
use deno_fs::FileSystemRc;
use deno_fs::FsPermissions;
use deno_fs::OpenOptions;
use deno_io::fs::FileResource;
use deno_permissions::PermissionsContainer;
use http_body_util::combinators::BoxBody;

use crate::CancelHandle;
use crate::CancelableResponseFuture;
use crate::FetchHandler;
use crate::ResourceToBodyAdapter;

/// An implementation which tries to read file URLs via `deno_fs::FileSystem`.
#[derive(Clone)]
pub struct FsFetchHandler;

impl FetchHandler for FsFetchHandler {
  fn fetch_file(
    &self,
    state: &mut OpState,
    url: &Url,
  ) -> (CancelableResponseFuture, Option<Rc<CancelHandle>>) {
    let cancel_handle = CancelHandle::new_rc();
    let Ok(path) = url.to_file_path() else {
      return (
        async move { Err(super::FetchError::NetworkError) }
          .or_cancel(&cancel_handle)
          .boxed_local(),
        Some(cancel_handle),
      );
    };
    let fs = state.borrow::<FileSystemRc>().clone();
    println!("fs: {fs:?}");
    let options = OpenOptions::read();
    let path = state
      .borrow_mut::<PermissionsContainer>()
      .check(true, &options, &path, "Deno.open()")
      .inspect_err(|p| println!("path_e: {p:?}"))
      .map(Cow::into_owned)
      .unwrap();

    let response_fut = async move {
      let file = fs
        .open_async(path.clone(), options, None)
        .await
        .map_err(|error| println!("open error: {error:?}-{:?}", &path))
        .unwrap();

      /*
            let file = fs
              .open_async(path, OpenOptions::read(), None)
              .await
              .inspect_err(|e| println!("ERROR: {e:?}"))
              .map_err(|_| super::FetchError::NetworkError)?;
      */
      let resource = Rc::new(FileResource::new(file, "".to_owned()));
      let body = BoxBody::new(ResourceToBodyAdapter::new(resource));
      let response = http::Response::new(body);
      Ok(response)
    }
    .or_cancel(&cancel_handle)
    .boxed_local();

    (response_fut, Some(cancel_handle))
  }
}

fn async_permission_check<P: FsPermissions + 'static>(
  state: Rc<RefCell<OpState>>,
  api_name: &'static str,
) -> impl AccessCheckFn {
  move |resolved, path, options| {
    let mut state = state.borrow_mut();
    let permissions = state.borrow_mut::<P>();
    permissions.check(resolved, options, path, api_name)
  }
}
